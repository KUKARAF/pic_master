# Region / pattern search benchmark

Measures how well a method finds the *same physical object* (task `region`) or the
*same pattern/texture* (task `pattern`) in other photos, instead of merely similar ones.

Dataset (not in git): `/var/home/rafa/Pictures/tastpictures/bench/` (override with
`--data-dir` or `$BENCH_DATA_DIR`): `manifest.json` plus every image under it except
`_review/` (that whole set is the corpus every query is ranked against). Cases are
`{id, task, synthetic, query_image, bbox, positives, hard_negatives}`; the query image
is removed from the ranking before scoring.

## Run

```
export LD_LIBRARY_PATH="$(dirname $(find /nix -name libGL.so.1 2>/dev/null | head -1)):$LD_LIBRARY_PATH"
cd /var/home/rafa/dev/pic_master/media_manager
/var/home/rafa/dev/pic_master/.venv/bin/python -I bench/harness.py --method baseline_clip_tiles \
    [--task region|pattern] [--real-only] [--json bench/results/NAME.json]
```

`--method all` runs every method in `bench/methods/`. The first run builds the index
(cached under `bench/.cache/<method>/`, git-ignored; it is rebuilt automatically when
the corpus changes), later runs only pay for queries.

## Add a method

Create `bench/methods/<anything>.py`:

```python
from harness import Method, top_k          # also: max_per_group, index_key
import numpy as np

class MyMethod(Method):
    name = 'my_method'        # unique; also the cache dir name
    kind = 'region'           # 'region' | 'pattern' | 'both'

    def warmup(self):         # optional: load models (excluded from timings)
        ...

    def build_index(self, image_paths, cache_dir):
        # image_paths: absolute paths of the whole corpus. Save features under cache_dir
        # if missing, then load them: it is only empty when the corpus changed.
        self.paths = list(image_paths)
        ...

    def query(self, image_path, bbox, k):
        # bbox = [x1, y1, x2, y2] in pixels of image_path. For kind='both', self.task is
        # 'region' or 'pattern' for the current call.
        scores = ...                       # one score per self.paths entry, higher = better
        return top_k(self.paths, scores, k)  # [(path, score)] best first

METHOD = MyMethod()           # or: def make(): return MyMethod()
```

Notes
- The harness asks for `k = len(corpus)` (full ranking, needed for mAP/recall@50); the
  latency numbers therefore include sorting everything. Returning fewer is allowed
  (missing positives count as never retrieved).
- `python -I` strips the script dir from `sys.path`; the harness re-adds `bench/` and the
  project root, so `from harness import ...` and `import media_manager...` work in methods.
- Index time/size come from the first (uncached) build and are stored in
  `.cache/<m>/_meta.json`; if you change a method's indexing, delete its cache dir.
- Prototypes needing an extra heavy dependency should say so in their report (install with `uv`).

## Metrics (per task; splits: all / real / synthetic)

- `R@k` recall@k: fraction of a case's positives within the top k, averaged over cases.
  With several positives per case R@1 cannot reach 1.0; `H@k` is the hit rate
  (any positive in the top k).
- `MRR` (first positive), `mAP` (all positives; unretrieved ones count as rank infinity).
- `HN@10`: fraction of cases with at least one hard negative in the top 10.
  `HN>pos`: fraction of cases where a hard negative ranks above the first positive.
  `conf@10` ("same-category confusion"): mean fraction of the top 10 that are hard negatives.
- Cost: index ms/image and KiB/image (first build), query latency p50/p95.

The baselines reuse the production code with the same semantics as `web.py`
(`api_region_search`: CLIP ViT-B/32 over `generate_tiles`, best-tile max cosine;
`api_pattern_search`: LBP+HSV+Gabor descriptor over the same tiles, best-tile max cosine).
Baseline timings are CPU (CLIP: `device=cpu`, no GPU locally).

## Baseline results

Corpus 1739 images; 66 region cases (41 real, 25 synthetic), 61 pattern cases (29 real, 32 synthetic).
CPU only, measured while the machine was heavily loaded (index times are pessimistic).

Region, `baseline_clip_tiles` (index ~5 s/img under load, 28 KiB/img; query p50 179 ms, p95 200 ms):

| split | R@1 | R@5 | R@10 | R@50 | H@1 | MRR | mAP | HN@10 | HN>pos | conf@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.144 | 0.571 | 0.720 | 0.821 | 0.682 | 0.778 | 0.646 | 0.470 | 0.227 | 0.083 |
| real | 0.201 | 0.718 | 0.885 | 0.925 | 0.976 | 0.982 | 0.868 | 0.268 | 0.000 | 0.032 |
| synthetic | 0.050 | 0.330 | 0.450 | 0.650 | 0.200 | 0.444 | 0.282 | 0.800 | 0.600 | 0.168 |

Pattern, `baseline_pattern_lbp` (index 87 ms/img, 8 KiB/img; query p50 19 ms, p95 34 ms):

| split | R@1 | R@5 | R@10 | R@50 | H@1 | MRR | mAP | HN@10 | HN>pos | conf@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.121 | 0.307 | 0.393 | 0.616 | 0.689 | 0.738 | 0.365 | 0.180 | 0.098 | 0.030 |
| real | 0.099 | 0.300 | 0.395 | 0.554 | 0.828 | 0.830 | 0.398 | 0.310 | 0.172 | 0.055 |
| synthetic | 0.141 | 0.312 | 0.391 | 0.672 | 0.562 | 0.656 | 0.334 | 0.062 | 0.031 | 0.006 |

Reading: the real region cases are same-landmark photos against other-landmark hard
negatives, which CLIP's category-level semantics already solve (H@1 0.98); the synthetic
cut-out-pasted-on-clutter cases are the discriminating ones (H@1 0.20, hard negatives
beat positives in 60% of cases). Full per-case results: `results/baseline_*.json`.

