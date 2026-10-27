"""Resumable, parallel open-domain candidate generation and solver labeling."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zipfile

from .io import save_json
from .generate import make_group
from .solver import check, label, problem


def source_manifest():
    import gramforge
    engine = Path(gramforge.__file__).parent
    files = sorted(engine.rglob('*.py')) + sorted(Path(__file__).parent.glob('*.json'))
    return {'files': {str(p.relative_to(engine)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'gramforge_base_commit': 'a83364169b0ba8dea6cce4ce17bb85ec57356246',
            'versions': {name: importlib.metadata.version(name) for name in ['z3-solver','pyarrow','funcy','exrex','easydict','anytree','nltk','pandas','numpy']}}


def archive_sources(root, sources):
    import gramforge
    engine = Path(gramforge.__file__).parent
    with zipfile.ZipFile(root / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources['files']):
            info = zipfile.ZipInfo('src/gramforge/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (engine / name).read_bytes())



def job(arguments):
    root, seed, index, hypotheses, executable, seconds = arguments
    path = Path(root) / 'groups' / f'{index:08d}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt') as file:
            return json.load(file)['counts']
    try:
        group = make_group(seed, index, hypotheses)
        counts = Counter()
        base = check(problem(group['background'], group['clauses']), executable, seconds)
        group['premise_check'] = base
        outcomes = []
        if base['status'] == 'Satisfiable':
            previous = None
            for hypothesis in group['hypotheses']:
                if previous is not None and hypothesis['formula'] == f'~({previous[0]})':
                    prior = previous[1]
                    result = prior | {'label': {'entailment':'contradiction','contradiction':'entailment'}.get(prior['label'], prior['label']),
                                      'positive': prior['negative'], 'negative': prior['positive'],
                                      'cache_note': 'Exact opposite of preceding hypothesis; polarities swapped'}
                else:
                    result = label(group['background'], group['clauses'], hypothesis['formula'], executable, seconds)
                previous = hypothesis['formula'], result
                counts[result['label']] += 1
                outcomes.append({'hypothesis': hypothesis, **result})
        else:
            counts['premises_' + base['status']] += 1
        group['outcomes'] = outcomes
        group['counts'] = dict(counts)
    except (ValueError, IndexError, KeyError) as error:
        group = {'index': index, 'seed': seed, 'error': str(error), 'counts': {'generation_error': 1}}
    temporary = path.with_suffix('.tmp')
    # Fixed gzip timestamp makes reruns independent of wall clock.
    with temporary.open('wb') as file:
        with gzip.GzipFile(fileobj=file, mode='wb', mtime=0, filename='') as compressed:
            compressed.write(json.dumps(group, sort_keys=True).encode())
    temporary.replace(path)
    return group['counts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups', type=int, default=1000)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hypotheses', type=int, default=12)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--vampire', default=str(Path.home() / 'extlibs/solver/vampire'))
    args = parser.parse_args()
    if args.groups < 1 or not 1 <= args.shards <= args.groups or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard configuration')
    root = Path(args.output)
    (root / 'groups').mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    sources['solver_sha256'] = hashlib.sha256(Path(args.vampire).read_bytes()).hexdigest()
    specification = {k: v for k, v in vars(args).items() if k not in ['workers', 'shard', 'shards', 'finalize']} | {'sources': sources}
    spec = root / 'specification.json'
    if spec.exists():
        previous = json.loads(spec.read_text())
        if {k:v for k,v in previous.items() if k != 'groups'} != {k:v for k,v in specification.items() if k != 'groups'} or args.groups < previous['groups']:
            raise ValueError('Changed inputs or source: use a fresh output directory')
        if args.groups > previous['groups']:
            save_json(spec, specification)
            (root / 'complete.json').unlink(missing_ok=True)
            for old_shard in root.glob('complete-shard*.json'):
                old_shard.unlink()
    else:
        save_json(spec, specification)
    if not (root / 'source.zip').exists():
        archive_sources(root, sources)
    if args.finalize:
        files = [root / 'groups' / f'{i:08d}.json.gz' for i in range(args.groups)]
        if not all(p.exists() for p in files):
            raise ValueError('Not all candidate groups are complete')
        counts = Counter()
        for path in files:
            with gzip.open(path, 'rt') as file:
                counts.update(json.load(file)['counts'])
        save_json(root / 'complete.json', {'groups_completed':args.groups, 'groups_target':args.groups, 'counts':dict(counts)})
        return
    counts = Counter()
    started = time.monotonic()
    jobs = [(str(root), args.seed, i, args.hypotheses, args.vampire, args.seconds) for i in range(args.shard, args.groups, args.shards)]
    with ProcessPoolExecutor(args.workers) as pool:
        pending = [pool.submit(job, item) for item in jobs]
        for done, future in enumerate(as_completed(pending), 1):
            counts.update(future.result())
            if done % 10 == 0 or done == len(jobs):
                status = {'groups_completed': done, 'groups_target': len(jobs), 'shard': args.shard, 'shards': args.shards,
                          'elapsed_seconds': round(time.monotonic() - started), 'counts': dict(counts)}
                save_json(root / ('progress.json' if args.shards == 1 else f'progress-shard{args.shard}.json'), status)
                print(json.dumps(status), flush=True)
    save_json(root / ('complete.json' if args.shards == 1 else f'complete-shard{args.shard}.json'), status)


if __name__ == '__main__':
    main()
