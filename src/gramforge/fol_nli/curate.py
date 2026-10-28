"""Notebook-derived curation over solver-labeled Gramforge candidates.

All shortening operates on paired clause objects. It preserves every proof
input; the exported result must subsequently pass the independent audit.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from transformers import AutoTokenizer

from .generate import digest
from .io import save_json, context_text
from .logic import parse
from .solver import premise_dependence
from .surface import stratum
from .build import archive_sources, source_manifest

LABELS = ['entailment', 'neutral', 'contradiction']


def normalized(node, bound=None):
    bound = {} if bound is None else bound
    op = node[0]
    if op == '~' and node[1][0] == '~':
        return normalized(node[1][1], bound)
    if op in ('!', '?'):
        variables = {v: f'V{len(bound) + i}' for i, v in enumerate(node[1])}
        return [op, list(variables.values()), normalized(node[2], bound | variables)]
    if op == 'atom':
        return [op, node[1], [bound.get(t, t) for t in node[2]]]
    if op in ('=', '!='):
        terms = [bound.get(t, t) for t in node[1:]]
        return [op, *sorted(terms)]
    parts = [normalized(child, bound) for child in node[1:]]
    if op in ('&', '|', '<=>', '<~>'):
        parts.sort(key=lambda p: json.dumps(p, sort_keys=True))
    return [op, *parts]


def features(row):
    p, h = row['premise_tptp'], row['hypothesis_tptp']
    return [len(p), len(h), len(row['premise']), len(row['hypothesis']),
            row['rule_concentration'], len(row['clauses']),
            *[s.count(op) for s in [p, h] for op in ['!', '?', '&', '|', '<=>', '=>', '<=', '<~>', '~', '=']],
            int(row['hypothesis'] in row['premise']), p.count(h)]


def select_clauses(clauses, support, budget, seed):
    # Setup/definition absence must not indicate whether an E/C proof uses it.
    required = support | {c['id'] for c in clauses if c['id'] == 'p0' or c.get('role') == 'definition'}
    if len(required) > budget:
        return None
    chosen = [c for c in clauses if c['id'] in required]
    unused = [c for c in clauses if c['id'] not in required]
    rng = random.Random(seed)
    rng.shuffle(unused)
    chosen += unused[:max(0, budget - len(chosen))]
    rng.shuffle(chosen)
    chosen.sort(key=lambda c: c['id'] != 'p0')
    return chosen


def match_surface(pools, count):
    """Equal label counts in each public surface bucket, up to rounding by one."""
    buckets = {label: defaultdict(list) for label in LABELS}
    for label, rows in pools.items():
        for row in rows:
            buckets[label][stratum(row)].append(row)
    keys = sorted(set().union(*(set(b) for b in buckets.values())))
    capacity = {key: min(len(buckets[label][key]) for label in LABELS) for key in keys}
    available = sum(capacity.values())
    common, remainder = divmod(count, len(LABELS))
    if available < common:
        raise ValueError(f'Insufficient matched surface buckets: {available} < {common} per label; grow candidate pool')
    quotas = {key: common * capacity[key] // available for key in keys} if common else dict.fromkeys(keys, 0)
    # Largest remainders allocate an exact total without changing label quotas.
    order = sorted(keys, key=lambda key: (-(common * capacity[key] % available), key)) if available else keys
    for key in order[:common - sum(quotas.values())]:
        quotas[key] += 1
    chosen = []
    for index, label in enumerate(LABELS):
        selected = [row for key in keys for row in buckets[label][key][:quotas[key]]]
        if index < remainder:
            used = {r['id'] for r in selected}
            extra = next((row for row in pools[label] if row['id'] not in used), None)
            if extra is None:
                raise ValueError(f'Insufficient {label} for rounding remainder; grow candidate pool')
            selected.append(extra)
        chosen.extend(selected)
    return chosen


def candidates(pool, max_premises, trace_cap, seed):
    seen, traces, rejected = set(), Counter(), Counter()
    rows = []
    final_seen = set()
    paths = sorted((Path(pool) / 'groups').glob('*.json.gz'))
    random.Random(seed).shuffle(paths)
    for path in paths:
        with gzip.open(path, 'rt') as file:
            group = json.load(file)
        if 'error' in group:
            continue
        formulas = {c['id']: normalized(parse(c['formula'])) for c in group['clauses']}
        statements = {digest(f) for f in formulas.values()}
        canonical_group = digest(sorted(formulas.values(), key=str))
        retained_groups = {}
        for outcome in group['outcomes']:
            label = outcome['label']
            if label not in LABELS:
                rejected[label] += 1; continue
            h = outcome['hypothesis']
            parsed_h = parse(h['formula'])
            normalized_h = normalized(parsed_h)
            # Both polarities must be checked: opposite-query generation can
            # reintroduce a statement excluded before the polarity expansion.
            if digest(normalized_h) in statements or digest(normalized(('~', parsed_h))) in statements:
                rejected['direct_premise_match'] += 1; continue
            dependency = None
            if label != 'neutral':
                dependency = outcome.get('dependency_check') or premise_dependence(group['background'], h['formula'], label)
                if dependency['status'] != 'dependent':
                    rejected['premise_' + dependency['status']] += 1; continue
            support = set(outcome['proof_inputs'])
            if label != 'neutral' and (not support or not outcome['proof_trace']):
                rejected['no_nontrivial_proof'] += 1; continue
            pair = digest([canonical_group, normalized_h])
            if pair in seen:
                rejected['duplicate'] += 1; continue
            seen.add(pair)
            trace_id = digest(outcome['proof_trace']) if outcome['proof_trace'] else 'neutral'
            if trace_id != 'neutral' and traces[trace_id] >= trace_cap:
                rejected['proof_trace_cap'] += 1; continue
            clauses = group['clauses']
            assert support <= {c['id'] for c in clauses}
            selected = select_clauses(clauses, support, max_premises, f'{seed}:{pair}')
            if selected is None:
                rejected['required_premises_exceed_budget'] += 1; continue
            assert support <= {c['id'] for c in selected}
            premise = '\n'.join(c['english'] for c in selected)
            # Person-domain and name conventions are part of the visible task.
            context = context_text(group['name_lexicon'], group['mentioned_names'])
            premise = context + '\n' + premise
            identifiers = tuple(sorted(c['id'] for c in selected))
            if identifiers not in retained_groups:
                retained_groups[identifiers] = digest([group['background'], sorted([formulas[i] for i in identifiers], key=str)])
            retained_group = retained_groups[identifiers]
            pair = digest([retained_group, normalized_h])
            if pair in final_seen:
                rejected['duplicate_after_shortening'] += 1; continue
            final_seen.add(pair)
            row = {'id': pair, 'group_id': canonical_group, 'full_group_id': canonical_group, 'retained_group_id': retained_group, 'context': context, 'premise': premise,
                   'hypothesis': h['english'], 'label': label,
                   'premise_tptp': '&'.join(f'({c["formula"]})' for c in selected),
                   'hypothesis_tptp': h['formula'], 'background': group['background'],
                   'clauses': selected, 'proof_inputs': sorted(support),
                   'proof_trace': outcome['proof_trace'], 'proof_trace_id': trace_id,
                   'proof': outcome.get('proof', outcome['proof_check']['output'] if outcome['proof_check'] else ''),
                   'proof_source': outcome.get('proof_source', 'legacy'),
                   'dependency_check': dependency,
                   'premise_unit_budget': min(len(clauses), max_premises),
                   'rule_concentration': group['rule_concentration'],
                   'source_group': path.name, 'source_group_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_index': group['index'],
                   'lexicon': group['lexicon'], 'name_lexicon': group['name_lexicon'], 'proposition_lexicon': group['proposition_lexicon']}
            rows.append(row); traces[trace_id] += 1
    # Connected components prevent leakage through either full or shortened
    # premises. Keep this grouping even for different hypotheses/paraphrases.
    parents = {r['group_id']: r['group_id'] for r in rows}
    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    retained = {}
    for row in rows:
        first = retained.setdefault(row['retained_group_id'], row['group_id'])
        a, b = find(first), find(row['group_id'])
        parents[max(a,b)] = min(a,b)
    for row in rows:
        row['group_id'] = find(row['group_id'])
    return rows, rejected


def curate(rows, size, seed, reference_bin=5, heuristic_size=1000):
    rng = random.Random(seed)
    rng.shuffle(rows)
    reserved = []
    for i, label in enumerate(LABELS):
        quota = heuristic_size // 3 + int(i < heuristic_size % 3)
        available = [r for r in rows if r['label'] == label]
        if len(available) < quota:
            raise ValueError(f'Insufficient {label} examples for reserved heuristic: {len(available)} < {quota}')
        reserved.extend(available[:quota])
    groups = {r['group_id'] for r in reserved}
    rows = [r for r in rows if r['group_id'] not in groups]
    if len({r['label'] for r in reserved}) != 3:
        raise ValueError('Heuristic training pool must contain all three labels')
    model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    model.fit(np.array([features(r) for r in reserved]), [r['label'] for r in reserved])
    probabilities = model.predict_proba(np.array([features(r) for r in rows]))
    predicted = model.classes_[probabilities.argmax(axis=1)]
    for row, guess, probability in zip(rows, predicted, probabilities.max(axis=1)):
        row['difficulty'] = float(probability * (1 if guess != row['label'] else -1))
    # Hash full canonical premise groups before shortening; related hypotheses
    # cannot cross splits, even when their retained distractors differ.
    buckets = defaultdict(list)
    for row in rows:
        value = int(digest([seed, row['group_id']])[:12], 16) / 16**12
        split = 'train' if value < .8 else 'validation' if value < .9 else 'test'
        buckets[split, row['label']].append(row)
    counts = {'train': int(size * .8), 'validation': int(size * .1)}
    counts['test'] = size - sum(counts.values())
    selected = {}
    caps = {}
    surface_counts = {}
    for split, count in counts.items():
        pools = {}
        for i, label in enumerate(LABELS):
            target = count // 3 + int(i < count % 3)
            pool = sorted(buckets[split, label], key=lambda r: (-r['difficulty'], r['id']))
            support_counts = Counter(len(r['proof_inputs']) for r in pool)
            if label != 'neutral':
                cap = support_counts[reference_bin]
                if not cap:
                    raise ValueError(f'Missing support bin {reference_bin} for {split}/{label}; grow the pool or explicitly change --reference-bin')
                keep, bins = [], Counter()
                for row in pool:
                    support_size = len(row['proof_inputs'])
                    if bins[support_size] < cap:
                        keep.append(row); bins[support_size] += 1
                pool = keep
                caps[f'{split}/{label}'] = cap
            if len(pool) < target:
                raise ValueError(f'Insufficient curated {split}/{label}: {len(pool)} < {target}; grow candidate pool')
            pools[label] = pool
        chosen = match_surface(pools, count)
        surface_counts[split] = {str(key): dict(Counter(r['label'] for r in chosen if stratum(r) == key))
                                 for key in sorted({stratum(r) for r in chosen})}
        rng.shuffle(chosen)
        selected[split] = chosen
    return selected, {'reserved_heuristic_rows': len(reserved), 'reserved_groups': sorted(groups),
                      'support_bin_reference': reference_bin, 'support_caps': caps,
                      'surface_stratum_counts': surface_counts}


def prepare(args):
    """Read and tokenize one immutable pool for one or more selection runs."""
    complete = Path(args.input) / 'complete.json'
    if not complete.exists():
        raise ValueError('Candidate pool is incomplete')
    expected = json.loads(complete.read_text())['groups_target']
    if len(list((Path(args.input) / 'groups').glob('*.json.gz'))) != expected:
        raise ValueError('Candidate group count does not match completion record')
    rows, rejected = candidates(args.input, args.max_premises, args.trace_cap, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision, local_files_only=True)
    filtered = []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        tokens = tokenizer([r['premise'] for r in chunk], [r['hypothesis'] for r in chunk], truncation=False)
        for row, ids in zip(chunk, tokens['input_ids']):
            if len(ids) <= args.max_tokens:
                row['tokens'] = len(ids); filtered.append(row)
            else:
                rejected['token_length'] += 1
    print(json.dumps({'stage':'prepared', 'rows':len(filtered),
                      'labels':dict(Counter(r['label'] for r in filtered)),
                      'support_sizes':{label:dict(Counter(len(r['proof_inputs']) for r in filtered if r['label'] == label))
                                       for label in LABELS}, 'rejected':dict(rejected)}), flush=True)
    return filtered, rejected


def export(args, splits, selection, rejected):
    root = Path(args.output)
    root.mkdir(parents=True)
    sources = source_manifest()
    archive_sources(root, sources)
    for split, selected in splits.items():
        pq.write_table(pa.Table.from_pylist(selected), root / f'{split}.parquet', compression='zstd')
    save_json(root / 'manifest.json', vars(args) | selection | {'rejected': dict(rejected),
              'curation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'curation_sources': sources,
              'pool_specification_sha256': hashlib.sha256((Path(args.input) / 'specification.json').read_bytes()).hexdigest(),
              'counts': {split: dict(Counter(r['label'] for r in selected)) for split, selected in splits.items()},
              'status': 'Requires fresh export verification; manifest alone is not a correctness certificate'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=100000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-premises', type=int, default=24)
    p.add_argument('--trace-cap', type=int, default=5000)
    p.add_argument('--reference-bin', type=int, default=5)
    p.add_argument('--heuristic-size', type=int, default=1000)
    p.add_argument('--max-tokens', type=int, default=500)
    p.add_argument('--tokenizer', default='sileod/deberta-v3-base-tasksource-nli')
    p.add_argument('--revision', default='3209a6ab012eab725e8f24547972f9aa133d1345')
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError('Use a fresh output directory')
    rows, rejected = prepare(args)
    splits, selection = curate(rows, args.size, args.seed, args.reference_bin, args.heuristic_size)
    export(args, splits, selection, rejected)


if __name__ == '__main__':
    main()
