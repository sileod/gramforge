import argparse
import pandas as pd
import random
import statistics
import time
from pathlib import Path

from gramforge.generate_sequential import generate_sequential
from gramforge.generate_sequential_opt import generate_sequential_opt
from gramforge.grammars import (
    FOL_grammar,
    arith_grammar,
    dyck_grammar,
    list_grammars,
    simple_english_grammar,
    tinypy_grammar,
)

ALGORITHMS = {
    'sequential': generate_sequential,
    'sequential_exp': generate_sequential_opt,
}

GRAMMAR_CONFIG = {
    arith_grammar.name: {'factory': arith_grammar, 'kwargs': {}, 'lang': 'py'},
    dyck_grammar.name: {'factory': dyck_grammar, 'kwargs': {'include_unicode': False}, 'lang': 'dyck'},
    simple_english_grammar.name: {'factory': simple_english_grammar, 'kwargs': {'cap': 3, 'questions': True}, 'lang': 'eng'},
    FOL_grammar.name: {'factory': FOL_grammar, 'kwargs': {'N_PREMS': 6, 'include_propositional': False, 'empty_room': False}, 'lang': 'eng'},
    tinypy_grammar.name: {'factory': tinypy_grammar, 'kwargs': {'level': '3.2'}, 'lang': 'py'},
}


def validate(prod, min_depth, max_depth, lang):
    if prod is None:
        return False, 'none'
    if not (min_depth <= prod.height <= max_depth):
        return False, f'height={prod.height}'
    if '#' in prod.render(lang):
        return False, 'placeholder'
    if not all(x.check() for x in [prod] + list(prod.descendants)):
        return False, 'constraint'
    return True, 'ok'


def instantiate_grammar(name):
    config = GRAMMAR_CONFIG[name]
    return config['factory'](**config['kwargs']), config['lang']


def run_case(grammar_name, algorithm_name, fn, *, seeds, min_depth, max_depth, max_steps, bushiness, k):
    grammar, lang = instantiate_grammar(grammar_name)
    rows = []
    for seed in seeds:
        random.seed(seed)
        t0 = time.perf_counter()
        result = fn(grammar.start(), min_depth=min_depth, max_depth=max_depth,
                    max_steps=max_steps, bushiness=bushiness, k=k)
        elapsed = time.perf_counter() - t0
        prod = result[0] if result else None
        ok, reason = validate(prod, min_depth, max_depth, lang)
        rows.append({
            'seed': seed,
            'success': bool(result),
            'valid': ok,
            'reason': reason,
            'height': None if prod is None else prod.height,
            'chars': 0 if prod is None else len(prod.render(lang)),
            'elapsed_s': elapsed,
        })

    successes = [row for row in rows if row['success']]
    return {
        'grammar': grammar_name,
        'algorithm': algorithm_name,
        'runs': len(rows),
        'min_depth': min_depth,
        'max_depth': max_depth,
        'max_steps': max_steps,
        'bushiness': bushiness,
        'k': k,
        'success_rate': sum(row['success'] for row in rows) / len(rows),
        'valid_rate': sum(row['valid'] for row in rows) / len(rows),
        'mean_height_success': statistics.mean(row['height'] for row in successes) if successes else None,
        'mean_chars_success': statistics.mean(row['chars'] for row in successes) if successes else None,
        'mean_elapsed_ms': 1000 * statistics.mean(row['elapsed_s'] for row in rows),
        'median_elapsed_ms': 1000 * statistics.median(row['elapsed_s'] for row in rows),
        'invalid_reasons': {reason: sum(row['reason'] == reason for row in rows) for reason in sorted({row['reason'] for row in rows if not row['valid']})},
    }


def compare_case(grammar_name, *, seeds, min_depth, max_depth, max_steps, bushiness, k, algorithms):
    results = {}
    for algorithm_name in algorithms:
        results[algorithm_name] = run_case(
            grammar_name,
            algorithm_name,
            ALGORITHMS[algorithm_name],
            seeds=seeds,
            min_depth=min_depth,
            max_depth=max_depth,
            max_steps=max_steps,
            bushiness=bushiness,
            k=k,
        )
    return results


def format_percent(x):
    return f'{100 * x:.1f}%'


def format_float(x):
    return '—' if x is None else f'{x:.3f}'


def render_markdown(results, *, runs, max_steps, bushiness, k, algorithms, cases):
    lines = []
    lines.append('# Generator benchmark comparison')
    lines.append('')
    lines.append(f'- grammars available via `list_grammars()`: {", ".join(list_grammars())}')
    lines.append(f'- algorithms compared: {", ".join(algorithms)}')
    lines.append(f'- runs per case: {runs}')
    lines.append(f'- max_steps: {max_steps}')
    lines.append(f'- bushiness: {bushiness}')
    lines.append(f'- k: {k}')
    lines.append(f'- cases: {", ".join(f"{lo}->{hi}" for lo, hi in cases)}')
    lines.append('')
    baseline_name = algorithms[0]
    candidate_name = algorithms[1]
    summary_rows = []
    for grammar_name, case_results in results:
        baseline = case_results[baseline_name]
        candidate = case_results[candidate_name]
        speedup = baseline['mean_elapsed_ms'] / candidate['mean_elapsed_ms'] if candidate['mean_elapsed_ms'] else None
        summary_rows.append({
            'Grammar': grammar_name,
            'Depth case': f"{baseline['min_depth']}->{baseline['max_depth']}",
            'Baseline ms': format_float(baseline['mean_elapsed_ms']),
            'Candidate ms': format_float(candidate['mean_elapsed_ms']),
            'Speedup': f"{format_float(speedup)}x",
            'Baseline success': format_percent(baseline['success_rate']),
            'Candidate success': format_percent(candidate['success_rate']),
            'Baseline valid': format_percent(baseline['valid_rate']),
            'Candidate valid': format_percent(candidate['valid_rate']),
            'Mean height (base/cand)': f"{format_float(baseline['mean_height_success'])} / {format_float(candidate['mean_height_success'])}",
        })
    lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))

    lines.append('')
    lines.append('## Invalid reason summary')
    lines.append('')
    reason_rows = []
    for grammar_name, case_results in results:
        for algorithm_name in algorithms:
            result = case_results[algorithm_name]
            reasons = ', '.join(f'{name}:{count}' for name, count in result['invalid_reasons'].items()) or 'none'
            reason_rows.append({
                'Grammar': grammar_name,
                'Depth case': f"{result['min_depth']}->{result['max_depth']}",
                'Algorithm': algorithm_name,
                'Invalid reasons': reasons,
            })
    lines.append(pd.DataFrame(reason_rows).to_markdown(index=False))

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grammars', nargs='*', default=['arith', 'dyck', 'english', 'fol', 'tinypy'])
    parser.add_argument('--algorithms', nargs='*', default=['sequential', 'sequential_exp'])
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--bushiness', type=float, default=0.8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--out', type=Path, default=ROOT / 'benchmarks' / 'generate_algorithm_comparison.md')
    args = parser.parse_args()

    missing = sorted(set(args.grammars) - set(GRAMMAR_CONFIG))
    if missing:
        raise ValueError(f'Unknown grammars: {missing}. Available benchmark configs: {sorted(GRAMMAR_CONFIG)}')
    missing_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if missing_algorithms:
        raise ValueError(f'Unknown algorithms: {missing_algorithms}. Available algorithms: {sorted(ALGORITHMS)}')
    if len(args.algorithms) != 2:
        raise ValueError('Markdown comparison currently expects exactly two algorithms.')

    seeds = list(range(args.runs))
    cases = [(6, 8), (8, 8)]
    results = []
    for grammar_name in args.grammars:
        for min_depth, max_depth in cases:
            results.append((grammar_name, compare_case(
                grammar_name,
                seeds=seeds,
                min_depth=min_depth,
                max_depth=max_depth,
                max_steps=args.max_steps,
                bushiness=args.bushiness,
                k=args.k,
                algorithms=args.algorithms,
            )))

    markdown = render_markdown(
        results,
        runs=args.runs,
        max_steps=args.max_steps,
        bushiness=args.bushiness,
        k=args.k,
        algorithms=args.algorithms,
        cases=cases,
    )
    args.out.write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
