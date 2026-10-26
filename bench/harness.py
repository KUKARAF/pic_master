"""Evaluation harness for region (instance) and pattern (texture) search methods.

    python -I bench/harness.py --method baseline_clip_tiles [--task region|pattern]
                               [--real-only] [--json out.json]

A method (see Method below) is plugged in by dropping bench/methods/<file>.py that
exposes METHOD (an instance) or make() (a factory). The harness builds its index over
the whole corpus (every image under the data dir except _review/), runs every manifest case
of the method's kind, and reports retrieval metrics split real / synthetic.

Manifest (<data dir>/manifest.json): {"cases": [...]} or a bare list; each case is
{id, task: 'region'|'pattern', synthetic, query_image, bbox [x1,y1,x2,y2], positives,
hard_negatives, notes}, paths relative to the data dir. The query image itself is never a
valid result: the harness removes it from every ranking before scoring.
"""
import abc
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BENCH_DIR)
# `python -I` drops the script dir and cwd from sys.path; put back what methods need
# (`from harness import Method`, `import media_manager...`).
for _p in (PROJECT_DIR, BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if __name__ == '__main__':
    sys.modules.setdefault('harness', sys.modules['__main__'])  # one Method class, not two

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
SKIP_DIRS = {'_review'}
# The dataset (images + manifest.json) lives outside the repo; override with --data-dir
# or $BENCH_DATA_DIR.
DATA_DIR = os.environ.get('BENCH_DATA_DIR', '/var/home/rafa/Pictures/tastpictures/bench')
CACHE_ROOT = os.path.join(BENCH_DIR, '.cache')
META_FILE = '_meta.json'
CUTOFFS = (1, 5, 10, 50)
CONFUSION_K = 10


class Method(abc.ABC):
    """A search method under test.

    name: unique id, also the cache dir name.
    kind: 'region' (instance retrieval of an object crop), 'pattern' (texture crop),
          or 'both'.
    """
    name = None
    kind = 'both'

    @abc.abstractmethod
    def build_index(self, image_paths, cache_dir):
        """Index `image_paths` (absolute). Persist everything under `cache_dir` so a
        repeat call with the same corpus loads from disk instead of recomputing; the
        harness owns the cache key (see index_key) and wipes cache_dir when the corpus
        changes, so files in it are always valid for the current corpus. The
        method need not exclude anything: the corpus list is already final."""

    @abc.abstractmethod
    def query(self, image_path, bbox, k):
        """Rank the indexed corpus for the crop `bbox` = [x1,y1,x2,y2] (pixels of
        `image_path`). Returns up to k [(path, score)], best first; higher = better.
        `task` flavour is chosen by the harness through `self.task` (set before the
        call) for methods with kind 'both'."""

    task = None

    def warmup(self):
        """Optional: load models here so one-off start-up cost is excluded from the
        index-time and query-latency numbers."""


def index_key(paths):
    """Stable key of a corpus (paths + sizes + mtimes); changes when the corpus does."""
    h = hashlib.sha1()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f'{p}\0{st.st_size}\0{int(st.st_mtime)}\n'.encode())
    return h.hexdigest()


def max_per_group(scores, starts):
    """Best score per image when rows are grouped contiguously (tiles of image i are
    rows starts[i]:starts[i+1]) -- the 'best-tile max' of the production endpoints."""
    return np.maximum.reduceat(scores, starts)


def top_k(paths, scores, k):
    """[(path, score)] of the k highest scores, best first."""
    scores = np.asarray(scores)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    idx = idx[np.argsort(-scores[idx], kind='stable')]
    return [(paths[i], float(scores[i])) for i in idx]


# ---------------------------------------------------------------- discovery / data

def discover_methods(methods_dir=None):
    """{name: Method} from methods/*.py (each exposes METHOD or make())."""
    methods_dir = methods_dir or os.path.join(BENCH_DIR, 'methods')
    found = {}
    for fn in sorted(os.listdir(methods_dir)):
        if not fn.endswith('.py') or fn.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(f'bench_methods_{fn[:-3]}',
                                                      os.path.join(methods_dir, fn))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = getattr(mod, 'METHOD', None) or (mod.make() if hasattr(mod, 'make') else None)
        if m is not None:
            found[m.name or fn[:-3]] = m
    return found


def corpus_paths(data_dir):
    out = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS:
                out.append(os.path.join(root, fn))
    return out


def load_cases(data_dir, manifest=None):
    """Manifest cases with every path made absolute."""
    with open(manifest or os.path.join(data_dir, 'manifest.json')) as f:
        data = json.load(f)
    cases = data['cases'] if isinstance(data, dict) else data
    absp = lambda p: os.path.normpath(os.path.join(data_dir, p))
    for c in cases:
        c['query_image'] = absp(c['query_image'])
        c['positives'] = [absp(p) for p in c.get('positives', [])]
        c['hard_negatives'] = [absp(p) for p in c.get('hard_negatives', [])]
        c['synthetic'] = bool(c.get('synthetic', False))
    # Dataset still being built / pruned: drop broken cases rather than crash mid-run.
    ok = [c for c in cases if os.path.isfile(c['query_image']) and c['positives']
          and all(os.path.isfile(p) for p in c['positives'])]
    if len(ok) < len(cases):
        print(f'warning: skipped {len(cases) - len(ok)} cases with missing files', file=sys.stderr)
    return ok


# ---------------------------------------------------------------------- metrics

def case_metrics(ranked, positives, hard_negatives):
    """Per-case numbers from a full ranking (query image already removed).
    Positives never retrieved count as rank infinity."""
    pos, hard = set(positives), set(hard_negatives)
    ranks = sorted(i + 1 for i, p in enumerate(ranked) if p in pos)
    out = {f'recall@{k}': sum(r <= k for r in ranks) / len(pos) for k in CUTOFFS}
    out.update({f'hit@{k}': float(bool(ranks) and ranks[0] <= k) for k in CUTOFFS})
    out['mrr'] = 1.0 / ranks[0] if ranks else 0.0
    out['ap'] = sum((i + 1) / r for i, r in enumerate(ranks)) / len(pos)
    top = ranked[:CONFUSION_K]
    out['hardneg_in_top10'] = float(any(p in hard for p in top))
    out['confusion@10'] = sum(p in hard for p in top) / CONFUSION_K
    first_hard = next((i for i, p in enumerate(ranked) if p in hard), None)
    out['hardneg_above_pos'] = float(first_hard is not None and (not ranks or first_hard < ranks[0] - 1))
    return out


def percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else float('nan')


# ------------------------------------------------------------------------- run

def prepare_index(method, paths):
    """Build (or load) the index; returns {seconds_per_image, bytes_per_image, cached}.
    Timing/size of the original build are kept in the cache dir's meta so a cached run
    still reports what indexing costs."""
    cache_dir = os.path.join(CACHE_ROOT, method.name)
    key = index_key(paths)
    meta_path = os.path.join(cache_dir, META_FILE)
    meta = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('key') != key:
            meta = None
    if meta is None:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    method.warmup()
    t0 = time.perf_counter()
    method.build_index(paths, cache_dir)
    elapsed = time.perf_counter() - t0
    if meta is None:
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(cache_dir) for f in fs)
        meta = {'key': key, 'seconds': elapsed, 'bytes': size}
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        cached = False
    else:
        cached = True
    n = max(1, len(paths))
    return {'index_s_per_image': meta['seconds'] / n, 'index_bytes_per_image': meta['bytes'] / n,
            'cached': cached, 'load_s': elapsed}


def run_cases(method, cases, paths, task):
    """Per-case metric dicts + latencies (s) for `task` cases."""
    method.task = task
    todo = [c for c in cases if c['task'] == task]
    results, lat = [], []
    if todo:   # warm-up: lazy model load / first-call allocations are not query latency
        method.query(todo[0]['query_image'], todo[0]['bbox'], 10)
    for c in todo:
        t0 = time.perf_counter()
        hits = method.query(c['query_image'], c['bbox'], len(paths))
        lat.append(time.perf_counter() - t0)
        ranked = [p for p, _ in hits if os.path.normpath(p) != c['query_image']]
        m = case_metrics(ranked, c['positives'], c['hard_negatives'])
        m.update(id=c['id'], synthetic=c['synthetic'])
        results.append(m)
    return results, lat


def summarize(results):
    keys = [k for k in results[0] if k not in ('id', 'synthetic')] if results else []
    return {'n': len(results), **{k: float(np.mean([r[k] for r in results])) for k in keys}}


def evaluate(method, data_dir, tasks=None, real_only=False, manifest=None):
    paths = corpus_paths(data_dir)
    cases = load_cases(data_dir, manifest)
    if real_only:
        cases = [c for c in cases if not c['synthetic']]
    index = prepare_index(method, paths)
    report = {'method': method.name, 'corpus_images': len(paths), 'index': index, 'tasks': {}}
    for task in ('region', 'pattern'):
        if (tasks and task not in tasks) or method.kind not in (task, 'both'):
            continue
        results, lat = run_cases(method, cases, paths, task)
        if not results:
            continue
        split = {'all': results,
                 'real': [r for r in results if not r['synthetic']],
                 'synthetic': [r for r in results if r['synthetic']]}
        report['tasks'][task] = {
            'metrics': {name: summarize(rs) for name, rs in split.items() if rs},
            'latency_p50_s': percentile(lat, 50), 'latency_p95_s': percentile(lat, 95),
            'per_case': results}
    return report


COLUMNS = [('n', 'n', '{:d}'), ('recall@1', 'R@1', '{:.3f}'), ('recall@5', 'R@5', '{:.3f}'),
           ('recall@10', 'R@10', '{:.3f}'), ('recall@50', 'R@50', '{:.3f}'),
           ('hit@1', 'H@1', '{:.3f}'), ('hit@10', 'H@10', '{:.3f}'), ('mrr', 'MRR', '{:.3f}'),
           ('ap', 'mAP', '{:.3f}'), ('hardneg_in_top10', 'HN@10', '{:.3f}'),
           ('hardneg_above_pos', 'HN>pos', '{:.3f}'), ('confusion@10', 'conf@10', '{:.3f}')]


def print_report(report):
    ix = report['index']
    print(f"\n== {report['method']}  corpus={report['corpus_images']} images  "
          f"index {ix['index_s_per_image'] * 1000:.1f} ms/img, "
          f"{ix['index_bytes_per_image'] / 1024:.1f} KiB/img"
          f"{' (cached build; load %.1fs)' % ix['load_s'] if ix['cached'] else ''}")
    for task, t in report['tasks'].items():
        print(f"\n[{task}]  query latency p50 {t['latency_p50_s'] * 1000:.0f} ms, "
              f"p95 {t['latency_p95_s'] * 1000:.0f} ms")
        print('  split      ' + ' '.join(f'{h:>7}' for _, h, _ in COLUMNS))
        for name, m in t['metrics'].items():
            print(f'  {name:<10} ' + ' '.join(f'{f.format(m[k] if k != "n" else int(m[k])):>7}'
                                             for k, _, f in COLUMNS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--method', required=True, help='method name, or "all"')
    ap.add_argument('--task', choices=('region', 'pattern'))
    ap.add_argument('--real-only', action='store_true', help='drop synthetic cases')
    ap.add_argument('--json', help='write the full report (incl. per-case) here')
    ap.add_argument('--data-dir', default=DATA_DIR, help='dataset dir (default: %(default)s)')
    ap.add_argument('--manifest', help='manifest path (default: <data-dir>/manifest.json)')
    args = ap.parse_args(argv)

    methods = discover_methods()
    names = sorted(methods) if args.method == 'all' else [args.method]
    reports = []
    for name in names:
        if name not in methods:
            sys.exit(f'unknown method {name!r}; available: {", ".join(sorted(methods))}')
        report = evaluate(methods[name], args.data_dir, [args.task] if args.task else None,
                          args.real_only, args.manifest)
        print_report(report)
        reports.append(report)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(reports[0] if len(reports) == 1 else reports, f, indent=1)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
