"""Verify final exported examples, split isolation and paired clause alignment."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
from functools import lru_cache
import json
from pathlib import Path

import pyarrow.parquet as pq

from .io import save_json, context_text
from .logic import parse, z3_check
from .solver import label, premise_dependence


@lru_cache(maxsize=128)
def source_group(path):
    content = Path(path).read_bytes()
    return hashlib.sha256(content).hexdigest(), json.loads(gzip.decompress(content))


def check_row(arguments):
    row, executable, seconds, independent, pool = arguments
    checksum, original = source_group(str(Path(pool) / "groups" / row["source_group"]))
    assert checksum == row["source_group_sha256"]
    original_clauses = {c["id"]: c for c in original["clauses"]}
    assert all(c == original_clauses[c["id"]] for c in row["clauses"])
    assert row["background"] == original["background"]
    assert any(o["hypothesis"]["formula"] == row["hypothesis_tptp"] and o["hypothesis"]["english"] == row["hypothesis"] and o["label"] == row["label"] for o in original["outcomes"])
    assert row["context"] == context_text(original["name_lexicon"], original["mentioned_names"])
    expected_premise = row['context'] + '\n' + '\n'.join(c['english'] for c in row['clauses'])
    assert row['premise'] == expected_premise, row['id']
    assert row['premise_tptp'] == '&'.join(f'({c["formula"]})' for c in row['clauses'])
    assert set(row['proof_inputs']) <= {c['id'] for c in row['clauses']}
    for formula in row['background'] + [c['formula'] for c in row['clauses']] + [row['hypothesis_tptp']]:
        parse(formula)
    verdict = label(row['background'], row['clauses'], row['hypothesis_tptp'], executable, seconds)
    if row['label'] in ('entailment', 'contradiction'):
        dependency = premise_dependence(row['background'], row['hypothesis_tptp'], row['label'], seconds)
        assert dependency['status'] == 'dependent', ('No certified premise dependence', row['id'], dependency)
    result = {'id': row['id'], 'label': row['label'], 'fresh_vampire': verdict['label'],
              'matches': verdict['label'] == row['label']}
    if independent:
        base = row['background'] + [c['formula'] for c in row['clauses']]
        positive = z3_check(base + [row['hypothesis_tptp']], seconds * 1000)
        negative = z3_check(base + [f'~({row["hypothesis_tptp"]})'], seconds * 1000)
        inferred = {('sat','unsat'):'entailment', ('unsat','sat'):'contradiction', ('sat','sat'):'neutral'}.get((positive,negative),'unresolved')
        result['independent_z3'] = inferred
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
    p.add_argument('--vampire', default=str(Path.home() / 'extlibs/solver/vampire'))
    args = p.parse_args()
    root = Path(args.build)
    seen, groups, retained, rows = set(), {}, {}, []
    for split in ['train', 'validation', 'test']:
        for row in pq.read_table(root / f'{split}.parquet').to_pylist():
            assert row['id'] not in seen, 'Duplicate canonical pair'
            seen.add(row['id'])
            previous = groups.setdefault(row['group_id'], split)
            assert previous == split, 'Premise group crosses splits'
            previous_retained = retained.setdefault(row['retained_group_id'], split)
            assert previous_retained == split, 'Shortened premise group crosses splits'
            rows.append(row)
    # Stable random-looking sample, fixed before checking any solver outcomes.
    sample = {r['id'] for r in sorted(rows, key=lambda r: hashlib.sha256(('independent:' + r['id']).encode()).hexdigest())[:args.independent_sample]}
    jobs = [(r, args.vampire, args.seconds, r['id'] in sample, args.pool) for r in rows]
    counts = Counter()
    independent = Counter()
    failures = []
    with ProcessPoolExecutor(args.workers) as pool:
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
              'passed': not failures, 'independent_unresolved_are_not_passes': True})
    if failures:
        raise RuntimeError(f'{len(failures)} exported rows unresolved or disagree')


if __name__ == '__main__':
    main()
