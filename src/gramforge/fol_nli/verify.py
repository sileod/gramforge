"""Verify final exported examples, split isolation and paired clause alignment."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path

import pyarrow.parquet as pq

from .io import save_json, context_text
from .logic import parse, z3_check
from .solver import label, premise_dependence
from .surface import stratum
from .build import source_manifest


def record_value(value):
    # Parquet unions optional struct fields, materializing absent keys as null.
    if isinstance(value, dict):
        return {k: record_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [record_value(v) for v in value]
    return value


def clause_matches(exported, original):
    fields = record_value(exported)
    return {'id','formula','english'} <= fields.keys() and all(
        key in original and value == record_value(original[key]) for key,value in fields.items())


def read_audit_rows(path):
    # The pinned source group already contains the authoritative derivation.
    # Avoid decoding/transferring a duplicate tree for every hypothesis.
    columns = ['id','label','group_id','retained_group_id','source_group','source_group_sha256',
               'background','hypothesis_tptp','hypothesis','context','premise','premise_tptp',
               'proof_inputs','premise_unit_budget']
    columns += ['clauses.list.element.' + field for field in ['id','formula','english','role']]
    return pq.ParquetFile(path).read(columns=columns).to_pylist()


@lru_cache(maxsize=128)
def source_group(path):
    content = Path(path).read_bytes()
    return hashlib.sha256(content).hexdigest(), json.loads(gzip.decompress(content))


def independent_check(base, hypothesis, seconds, retry_seconds=0):
    formulas = [base + [hypothesis], base + [f'~({hypothesis})']]
    statuses = [z3_check(f,seconds * 1000) for f in formulas]
    attempts = [{'seconds':seconds,'random_seed':0,'positive':statuses[0],'negative':statuses[1]}]
    if 'unknown' in statuses and retry_seconds > seconds:
        statuses = [z3_check(f,retry_seconds * 1000,random_seed=42,ematching=False) if status == 'unknown' else status
                    for f,status in zip(formulas,statuses)]
        attempts.append({'seconds':retry_seconds,'random_seed':42,'ematching':False,
                         'positive':statuses[0],'negative':statuses[1]})
    inferred = {('sat','unsat'):'entailment', ('unsat','sat'):'contradiction',
                ('sat','sat'):'neutral'}.get(tuple(statuses),'unresolved')
    return inferred, attempts


def check_row(arguments):
    row, executable, seconds, independent, pool, retry_seconds = arguments
    checksum, original = source_group(str(Path(pool) / "groups" / row["source_group"]))
    assert checksum == row["source_group_sha256"]
    original_clauses = {c["id"]: c for c in original["clauses"]}
    assert all(clause_matches(c,original_clauses[c['id']]) for c in row['clauses'])
    mandatory = {c['id'] for c in original['clauses'] if c['id'] == 'p0' or c.get('role') == 'definition'}
    assert mandatory <= {c['id'] for c in row['clauses']}, 'Setup/definition omitted during shortening'
    assert len(row['clauses']) == row['premise_unit_budget'], 'Label-dependent clause budget'
    assert row["background"] == original["background"]
    assert any(o["hypothesis"]["formula"] == row["hypothesis_tptp"] and o["hypothesis"]["english"] == row["hypothesis"] and o["label"] == row["label"] for o in original["outcomes"])
    assert row["context"] == context_text(original["name_lexicon"], original["mentioned_names"])
    expected_premise = row['context'] + '\n' + '\n'.join(c['english'] for c in row['clauses'])
    assert row['premise'] == expected_premise, row['id']
    assert row['premise_tptp'] == '&'.join(f'({c["formula"]})' for c in row['clauses'])
    assert set(row['proof_inputs']) <= {c['id'] for c in row['clauses']}
    for formula in row['background'] + [c['formula'] for c in row['clauses']] + [row['hypothesis_tptp']]:
        parse(formula)
    # Labels require fresh checks of both polarities, not another extraction
    # of proof support already retained from generation.
    verdict = label(row['background'], row['clauses'], row['hypothesis_tptp'], executable, seconds, recover_proof=False)
    if row['label'] in ('entailment', 'contradiction'):
        dependency = premise_dependence(row['background'], row['hypothesis_tptp'], row['label'], seconds)
        assert dependency['status'] == 'dependent', ('No certified premise dependence', row['id'], dependency)
    result = {'id': row['id'], 'label': row['label'], 'fresh_vampire': verdict['label'],
              'matches': verdict['label'] == row['label']}
    if independent:
        base = row['background'] + [c['formula'] for c in row['clauses']]
        inferred, attempts = independent_check(base,row['hypothesis_tptp'],seconds,retry_seconds)
        result['independent_z3'] = inferred
        result['independent_attempts'] = attempts
        if inferred != 'unresolved':
            assert inferred == row['label'], result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build')
    p.add_argument('--pool', required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--seconds', type=int, default=5)
    p.add_argument('--independent-sample', type=int, default=1000)
    p.add_argument('--independent-retry-seconds', type=int, default=0)
    p.add_argument('--vampire', default=str(Path.home() / 'extlibs/solver/vampire'))
    args = p.parse_args()
    root = Path(args.build)
    seen, groups, retained, rows = set(), {}, {}, []
    for split in ['train', 'validation', 'test']:
        split_rows = read_audit_rows(root / f'{split}.parquet')
        buckets = defaultdict(Counter)
        for row in split_rows:
            buckets[stratum(row)][row['label']] += 1
        for key,counts in buckets.items():
            assert max(counts.get(label, 0) for label in ['entailment','neutral','contradiction']) - min(
                counts.get(label, 0) for label in ['entailment','neutral','contradiction']) <= 1, ('Surface imbalance', split, key, counts)
        for row in split_rows:
            assert row['id'] not in seen, 'Duplicate canonical pair'
            seen.add(row['id'])
            previous = groups.setdefault(row['group_id'], split)
            assert previous == split, 'Premise group crosses splits'
            previous_retained = retained.setdefault(row['retained_group_id'], split)
            assert previous_retained == split, 'Shortened premise group crosses splits'
            rows.append(row)
    # Stable random-looking sample, fixed before checking any solver outcomes.
    sample = {r['id'] for r in sorted(rows, key=lambda r: hashlib.sha256(('independent:' + r['id']).encode()).hexdigest())[:args.independent_sample]}
    jobs = [(r, args.vampire, args.seconds, r['id'] in sample, args.pool, args.independent_retry_seconds) for r in rows]
    counts = Counter()
    independent = Counter()
    failures = []
    # Arrow/solver imports may already have threads; do not inherit their locks.
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn')) as pool:
        with (root / 'verification.jsonl').open('w') as file:
            for result in pool.map(check_row, jobs, chunksize=8):
                file.write(json.dumps(result) + '\n')
                counts[result['fresh_vampire']] += 1
                if 'independent_z3' in result:
                    independent[result['independent_z3']] += 1
                if not result['matches']:
                    failures.append(result)
    save_json(root / 'verification.json', {'rows': len(rows), 'vampire': dict(counts),
              'independent_z3': dict(independent), 'failures': failures,
              'passed': not failures, 'independent_unresolved_are_not_passes': True,
              'independent_retry_seconds':args.independent_retry_seconds,
              'verification_sources': source_manifest()})
    if failures:
        raise RuntimeError(f'{len(failures)} exported rows unresolved or disagree')


if __name__ == '__main__':
    main()
