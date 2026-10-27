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

LABELS = ['entailment', 'neutral', 'contradiction']


def normalized(node, bound=None):
    bound = {} if bound is None else bound
    op = node[0]
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


def candidates(pool, max_premises, trace_cap, seed):
    seen, traces, rejected = set(), Counter(), Counter()
    rows = []
    final_seen = set()
    for path in sorted((Path(pool) / 'groups').glob('*.json.gz')):
        with gzip.open(path, 'rt') as file:
            group = json.load(file)
        if 'error' in group:
            continue
        canonical_group = digest(sorted([normalized(parse(c['formula'])) for c in group['clauses']], key=str))
        for outcome in group['outcomes']:
            label = outcome['label']
            if label not in LABELS:
                rejected[label] += 1; continue
            support = set(outcome['proof_inputs'])
            if label != 'neutral' and (not support or not outcome['proof_trace']):
                rejected['no_nontrivial_proof'] += 1; continue
            h = outcome['hypothesis']
            pair = digest([canonical_group, normalized(parse(h['formula']))])
            if pair in seen:
                rejected['duplicate'] += 1; continue
            seen.add(pair)
            trace_id = digest(outcome['proof_trace']) if outcome['proof_trace'] else 'neutral'
            if trace_id != 'neutral' and traces[trace_id] >= trace_cap:
                rejected['proof_trace_cap'] += 1; continue
            clauses = group['clauses']
            assert support <= {c['id'] for c in clauses}
            rng = random.Random(f'{seed}:{pair}')
            used = [c for c in clauses if c['id'] in support]
            unused = [c for c in clauses if c['id'] not in support]
            rng.shuffle(unused)
            selected = used + unused[:max(0, max_premises - len(used))]
            rng.shuffle(selected)
            selected.sort(key=lambda c: c['id'] != 'p0')
            assert support <= {c['id'] for c in selected}
            premise = '\n'.join(c['english'] for c in selected)
            # Person-domain and name conventions are part of the visible task.
            context = context_text(group['name_lexicon'], group['mentioned_names'])
            premise = context + '\n' + premise
            retained_group = digest([group['background'], sorted([normalized(parse(c['formula'])) for c in selected], key=str)])
            pair = digest([retained_group, normalized(parse(h['formula']))])
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
    for split, count in counts.items():
        chosen = []
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
            chosen.extend(pool[:target])
        rng.shuffle(chosen)
        selected[split] = chosen
    return selected, {'reserved_heuristic_rows': len(reserved), 'reserved_groups': sorted(groups),
                      'support_bin_reference': reference_bin, 'support_caps': caps}


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
    root = Path(args.output)
    if root.exists():
        raise ValueError('Use a fresh output directory')
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
    splits, selection = curate(filtered, args.size, args.seed, args.reference_bin, args.heuristic_size)
    root.mkdir(parents=True)
    for split, selected in splits.items():
        pq.write_table(pa.Table.from_pylist(selected), root / f'{split}.parquet', compression='zstd')
    save_json(root / 'manifest.json', vars(args) | selection | {'rejected': dict(rejected),
              'curation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pool_specification_sha256': hashlib.sha256((Path(args.input) / 'specification.json').read_bytes()).hexdigest(),
              'counts': {split: dict(Counter(r['label'] for r in selected)) for split, selected in splits.items()},
              'status': 'Requires fresh export verification; manifest alone is not a correctness certificate'})


if __name__ == '__main__':
    main()
