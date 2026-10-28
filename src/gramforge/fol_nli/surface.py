"""Text-only shortcut probes, with premise groups kept out of training folds."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import context_text, save_json

LABELS = ['entailment', 'neutral', 'contradiction']


def stratum(row):
    """Coarse public-text features matched across labels during curation."""
    premise, hypothesis = row['premise'].lower(), row['hypothesis'].lower()
    return (int('definition:' in premise), int(bool(re.search(r'\broom\b', premise))),
            int('room occupant' in premise),
            int(hypothesis.startswith('it is not the case that')),
            int(len(premise.split()) > 120), int(len(hypothesis.split()) > 12))


def formatting(row):
    result = []
    for text in [row['premise'], row['hypothesis']]:
        lower = text.lower()
        result += [len(text), len(text.split()), text.count('\n'),
                   text.count('“'), text.count('”'), text.count('"'),
                   lower.count('definition:'), lower.count('in the room'),
                   lower.count('outside the room'), lower.count('room occupant'),
                   int(bool(re.search(r'\broom\b', lower))), int('definition:' in lower),
                   int(lower.startswith('it is not the case that'))]
    return result


def diagnose(rows, seed=42):
    labels = np.array([r['label'] for r in rows])
    groups = np.array([r['group_id'] for r in rows])
    counts = Counter(labels)
    result = {'rows': len(rows), 'premise_groups': len(set(groups)),
              'labels': dict(counts), 'chance_balanced_accuracy': 1 / len(LABELS),
              'note': 'Scores detect associations, not proof of a causal leak. Inspect held-out groups, not random row splits.',
              'by_label': {}}
    for label in LABELS:
        subset = [r for r in rows if r['label'] == label]
        if subset:
            result['by_label'][label] = {
                'mean_premise_words': float(np.mean([len(r['premise'].split()) for r in subset])),
                'mean_hypothesis_words': float(np.mean([len(r['hypothesis'].split()) for r in subset])),
                'definition_presence': float(np.mean(['definition:' in r['premise'].lower() for r in subset])),
                'room_presence': float(np.mean([bool(re.search(r'\broom\b', r['premise'], re.I)) for r in subset])),
                'mean_in_the_room_mentions': float(np.mean([r['premise'].lower().count('in the room') for r in subset])),
                'negated_hypothesis_wrapper': float(np.mean([r['hypothesis'].lower().startswith('it is not the case that') for r in subset]))}
    if min(counts.get(label, 0) for label in LABELS) < 3 or len(set(groups)) < 3:
        return result | {'status': 'insufficient_groups_or_labels'}
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(rows)), labels, groups))
    result['probes'] = {}
    for name in ['formatting', 'hypothesis_only', 'premise_only']:
        values = np.asarray([formatting(r) for r in rows]) if name == 'formatting' else np.asarray(
            [r['hypothesis' if name == 'hypothesis_only' else 'premise'] for r in rows])
        predictions = np.empty(len(rows), dtype=object)
        scores = []
        for train, test in folds:
            assert not set(groups[train]) & set(groups[test])
            transform = StandardScaler() if name == 'formatting' else TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
            model = make_pipeline(transform, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))
            model.fit(values[train], labels[train])
            predictions[test] = model.predict(values[test])
            scores.append(float(balanced_accuracy_score(labels[test], predictions[test])))
        result['probes'][name] = {'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
                                 'fold_balanced_accuracies': scores,
                                 'confusion_matrix': confusion_matrix(labels, predictions, labels=LABELS).tolist()}
    return result | {'status': 'measured', 'folds': 3, 'label_order': LABELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool')
    source.add_argument('--dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = []
    if args.pool:
        for path in sorted((Path(args.pool) / 'groups').glob('*.json.gz')):
            with gzip.open(path, 'rt') as file:
                group = json.load(file)
            if 'error' in group:
                continue
            premise = context_text(group['name_lexicon'], group['mentioned_names']) + '\n' + '\n'.join(c['english'] for c in group['clauses'])
            rows.extend({'premise': premise, 'hypothesis': o['hypothesis']['english'],
                         'label': o['label'], 'group_id': group['group_id']}
                        for o in group['outcomes'] if o['label'] in LABELS)
    else:
        for split in ['train', 'validation', 'test']:
            rows.extend(pq.read_table(Path(args.dataset) / f'{split}.parquet',
                        columns=['premise', 'hypothesis', 'label', 'group_id']).to_pylist())
    report = diagnose(rows, args.seed)
    report.update({'source': vars(args), 'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
