"""Grammar productivity, structural coverage and retained-pool diagnostics."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from .grammar import make_grammar
from .io import save_json


def inventory():
    grammar = make_grammar()
    productive = set()
    while True:
        previous = productive.copy()
        for rule in grammar._instances:
            if set(rule.args) <= productive:
                productive.add(rule.name)
        if productive == previous:
            break
    reachable = {'start'}
    while True:
        previous = reachable.copy()
        for rule in grammar._instances:
            if rule.name in reachable:
                reachable.update(rule.args)
        if reachable == previous:
            break
    return {'rules': [{'index': r.index, 'signature': r.signature,
                       'productive': set(r.args) <= productive,
                       'reachable': r.name in reachable,
                       'formula': r.templates['tptp'] if isinstance(r.templates['tptp'],str) else 'callable: ' + r.templates['tptp'].__qualname__}
                      for r in grammar._instances],
            'unproductive_symbols': sorted(reachable - productive)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pool')
    args = p.parse_args()
    counts, lengths, supports, rules = Counter(), Counter(), Counter(), Counter()
    for path in sorted((Path(args.pool) / 'groups').glob('*.json.gz')):
        with gzip.open(path, 'rt') as file:
            group = json.load(file)
        counts.update(group['counts'])
        if 'error' in group:
            continue
        lengths[len(group['clauses']) - 1] += 1
        stack = [group['derivation']]
        while stack:
            node = stack.pop(); rules[node['signature']] += 1; stack.extend(node['children'])
        for outcome in group['outcomes']:
            if outcome['label'] in ['entailment','contradiction']:
                supports[len(outcome['proof_inputs'])] += 1
    save_json(Path(args.pool) / 'coverage.json', inventory() | {'counts': dict(counts),
              'premise_unit_counts': dict(lengths), 'proof_support_sizes': dict(supports),
              'observed_rules': dict(rules)})
    print(json.dumps({'counts': dict(counts), 'lengths': dict(lengths), 'supports': dict(supports)}, indent=2))


if __name__ == '__main__':
    main()
