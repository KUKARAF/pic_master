# Region (R) and Pattern (P) search: why they fail, options, shortlist

Research agent output. No source code was changed. Date: 2026-10-26.
Scope: photo-view buttons "region" (CLIP tiles) and "pattern" (LBP+HSV+Gabor).
Target: instance / pattern-instance retrieval over ~100k images, <2 s per query on 4 CPU cores, optional remote GPU worker.

Evidence tags: **[code]** read in this repo, **[measured]** run on this box (4 cores, 7 GB RAM, heavily loaded by other jobs, so timings are upper bounds), **[verified]** checked against PyPI / HF API / GitHub README today, **[memory]** my recollection of literature, not re-checked (Firecrawl credits were exhausted, so I could not pull papers; cited URLs are the canonical ones).

---

## 0. Executive summary

1. Both features answer a different question than the one the user asks. CLIP tiles measure "same kind of scene content"; the pattern descriptor measures "similar global colour/micro-texture mix of a very coarse tile". Neither has any notion of "the same physical thing", and neither localises the object inside the candidate photo.
2. There are also concrete bugs that would hurt even a good model. The worst: **no EXIF orientation handling anywhere** (`grep exif_transpose` returns nothing) while the browser draws the EXIF-rotated image and sends boxes in that rotated frame. Every portrait phone photo gets the wrong crop in `region-search`. See section 1.3.
3. The repo already contains most of the machinery for a real fix: a DINOv2-with-registers `ObjectEncoder`, an `object_region_embeddings` table, and an XFeat/LighterGlue/SIFT + MAGSAC `ObjectMatcher` that does geometric verification (used by location object-search). The recommended path is to reuse those for R and add a geometric / dense-token re-rank, not to invent new infrastructure.
4. Recommended shortlist (details in section 4):
   - R1: DINOv2 multi-scale window pooling (prefilter) + RootSIFT/XFeat RANSAC re-rank. This is the hybrid and the best expected accuracy per cost.
   - R2: DINOv2 dense patch tokens with late-interaction (MaxSim) matching over a stored, PCA-compressed token grid. Handles partial / small objects without any detector.
   - P1: Same dense-token MaxSim, but multi-scale query pyramid, restricted to the texture case.
   - P2: Periodicity-aware pipeline: autocorrelation unit-cell estimate, then normalised multi-scale NCC template verification of candidates (exactly repeating patterns), with RootSIFT+RANSAC for aperiodic "spots".
   - H: One shared two-stage hybrid (cheap ANN prefilter on pooled DINOv2 windows, then geometric / token re-rank). R1 and P2 share the prefilter.

---

## 1. Why the current methods fail

### 1.1 Region search (CLIP tiles) [code: tile_index.py, web.py `api_region_search`, indexer.py]

Pipeline: index = 1 whole + 2x2 + 3x3 overlapping crops per image, each CLIP ViT-B/32 (openai) embedded (512-d, fp32). Query = drawn box cropped, CLIP embedded, cosine against every tile, per-image max, top 50.

Mechanisms of failure, most important first:

1. **Wrong invariance.** CLIP ViT-B/32 openai is trained by image-text contrast on captions ("a gold bracelet"). Its embedding is deliberately invariant to everything that distinguishes two bracelets of one category and highly sensitive to the category. Cosine between two different bracelets is typically higher than between one bracelet and its own crop under a different viewpoint. The symptom "marking my bracelet returns any bracelet" is the model working as designed. No tuning of the current pipeline fixes this.
2. **Tiles never align with the query.** The grid is fixed: tile sizes are 2/3 and 1/2 of each image dimension (the docstring says "half" for both; for n=2 the tile is `2/(n+1) = 2/3`). The smallest tile is therefore 25% of the image area, at 14 fixed positions. A bracelet or a flower that occupies 2-10% of the frame is never the dominant content of any tile, so its tile embedding is a scene embedding. The query is a tight crop (object fills 100% of the input), the target tile is mostly background, and cosine is low for the true match and about equal to the cosine of any same-category scene. Max-over-tiles then picks arbitrary scene similarity.
3. **No scale search at the query side either.** The query is embedded once at its own scale. Instance retrieval needs the index to contain windows at the scale at which the object appears, or the query to be matched against pooled features at multiple scales.
4. **Aspect ratio / centre crop.** `embed_pil_images` uses open_clip's default `preprocess` = Resize(224, shortest side) + CenterCrop(224). A wide query box (say 400x100) is reduced to its central 100x100 square; the two ends are discarded. Tiles of a 3:2 photo lose 1/6 of the long side each side. (`ObjectEncoder._preprocess` does the opposite: it squashes to 224x224 with `resize((224,224))`, distorting aspect ratio; for DINOv2 patches this changes apparent shape and hurts matching.)
5. **Resolution.** 224 px input. A tile of a 12 MP photo is downscaled 10-20x, which removes the fine detail (engraving, stitching, grain) that distinguishes instances.
6. **EXIF rotation bug (see 1.3).** Query crop and tile content are in the wrong orientation for rotated photos.
7. **Scoring/IO issues that make it slow and noisy:**
   - Each query re-reads every tile row from SQLite (`iter_tile_embeddings`): at 100k images that is about 1.4M blobs and about 2.9 GB through the sqlite->bytes->numpy path per query, plus a Python `for idx in range(scores.shape[0])` loop over 1.4M scores with `float()` and dict updates. This is seconds to tens of seconds, not <2 s. A resident fp16 matrix plus `np.maximum.reduceat` / a per-file segment max removes it.
   - Near-duplicates and bursts: the best-tile-per-file max has no de-duplication, so one scene shot 12 times fills the top 50.
   - Partial index is silently accepted: `count_tiled_files() > 0` switches to tile mode even if only a few files are tiled; the UI never shows `mode`, so the user cannot tell whether results come from tiles or the whole-image fallback. Tile rows store a `model` column but search does not filter on it, so a CLIP-model change mixes incompatible spaces.
   - The `model_id` of the query encoder and the tile encoder are never compared; the docstring of the endpoint still says "rank the library's whole-image embeddings" while tile mode is what runs.
8. **Result is not localised.** The endpoint returns only file ids; the best tile bbox is discarded, so the user cannot see why a photo matched (and false positives are unexplainable).

### 1.2 Pattern search (LBP59 + HSV72 + Gabor16, cosine) [code: pattern_descriptor.py, web.py `api_pattern_search`, worker_server.py `handle_pattern`]

1. **Scale mismatch between query and index (the dominant bug).** Query: a small user box (e.g. 80x80 px of the wall) resized to 128x128: it is *up*-sampled, so LBP sees interpolated, smooth, near-monotone gradients, producing a histogram heavy in specific uniform codes. Index: each tile is 1/4 to 4/9 of a whole photo (e.g. 1500x1000 px) *down*-sampled 8-12x to 128x128 with INTER_AREA. At that scale a wall's micro-texture is averaged away and the tile contains whole scene structure. The descriptors live in different regimes (different frequency band, different content mixture), so cosine is meaningless between them.
2. **Tile = mixture.** A pattern covering 5% of an image contributes 5% of the histogram mass. The whole-tile histogram is dominated by whatever else is in it (sky, furniture, faces). Max-over-14-tiles does not fix this; there is no tile at the pattern's scale or position.
3. **Weighting inside the concatenated vector.** The three blocks are concatenated then L2-normalised jointly. Gabor block is L2-normalised to 1.0 by itself, while the LBP and HSV blocks are L1-normalised histograms (L2 norm around 0.3-0.7). So the Gabor block carries a large share of the energy. The Gabor means/stds are all non-negative responses of kernels that are not zero-mean (`cv2.getGaborKernel` with psi=0 has a DC component), so after normalisation every image maps to nearly the same direction: a large near-constant offset of cosine (scores crowd into e.g. 0.85-0.99) that carries little information. The HSV block (72 bins, 8x8 hue-sat) is the only strongly discriminative block and it is a *colour* descriptor, so the ranking is mostly "similar colour palette".
4. **Cosine on histograms.** Dot product of L2-normalised raw histograms is dominated by the largest bins (LBP bin 58 "non-uniform", value-histogram peaks). The standard remedies, Hellinger (sqrt then L2) or chi-square, are not used.
5. **No invariance that the use case requires.**
   - Rotation: LBP is the plain 8-neighbour uniform LBP (not rotation-invariant `riu2`); Gabor is bank-of-4-orientations energy without rotation pooling/sorting. Camera roll or a wallpaper seen on a different wall breaks it.
   - Scale: single scale 128 px, single LBP radius 1, two Gabor wavelengths (4, 8 px).
   - Lighting / white balance: HSV histogram shifts with illumination.
   - Perspective: none. A wall in perspective is a different texture frequency left to right.
6. **Not an instance descriptor.** 147 numbers of coarse statistics cannot distinguish two different floral wallpapers of similar palette; they are *category-level texture statistics*. For a specific stain, crack, brick arrangement, or print motif there is nothing in the descriptor that encodes arrangement (spatial layout), which is what makes the pattern "that one".
7. **Worker vs local divergence and cost** [code]:
   - Same module on both sides (good), but `cv2.imdecode(..., IMREAD_COLOR)` applies EXIF orientation while the tile path in `tile_index`/CLIP uses PIL without it. So pattern bboxes are in the rotated frame and CLIP tile bboxes in the raw-sensor frame: the two indexes disagree about where things are. For queries, pattern search happens to be right and region search wrong.
   - Each query and each index item ships the **full original file bytes** to the worker over RNS (`wc.pattern_query(image_bytes, ...)`), for a computation that takes ~10 ms. For the 100k index build that is the entire library over the link for no benefit; CPU-only work needs no worker.
   - `ALGO_TAG` is stored but the search only skips on *dimension* mismatch (`matrix.shape[1] != q.shape[0]`), not on tag: after a same-dimension algorithm change, stale descriptors are silently compared.
   - `get_unpattern_indexed_files` filters `hidden = 0` but `get_untiled_files` does not: the two indexes cover different sets.
8. **UI** [code: static/app.js `wireBoxDraw`]: bbox = CSS box scaled by `naturalWidth/clientWidth`, so it is in the *displayed (EXIF-rotated)* frame. Region and pattern calls send only `{bbox}`; no scale hint, no mask, no "this is periodic" flag. The user cannot mark multiple examples, negatives, or "same scale only".

### 1.3 Cross-cutting concrete defects

| # | Defect | Where | Effect |
|---|---|---|---|
| 1 | No `ImageOps.exif_transpose` anywhere; browser renders `/image/{id}` (raw bytes) rotated; bbox is in the rotated frame | web.py `api_region_search`, `tile_index.embed_tiles_for_file`, object_index | Wrong crop for every EXIF-rotated photo (orientation 3, 6, 8: most phone portrait shots). Tiles of rotated photos embed sideways content. |
| 2 | CLIP center-crop of non-square crops | `embed_pil_images` | Drops ends of wide/tall boxes |
| 3 | ObjectEncoder squashes to square | `_preprocess` | Aspect distortion, hurts DINOv2 instance matching |
| 4 | Scoring loop in Python, full SQLite scan per query | `api_region_search`, `api_pattern_search` | Seconds to minutes at 100k images; GIL-bound |
| 5 | Per-file max without dedup / no localisation returned | both | Duplicate floods, unexplainable hits |
| 6 | Pattern index covers `hidden=0` only; tile index all files | database.py | Inconsistent coverage |
| 7 | Algo / model tag not enforced at query | both | Silent mixed spaces |
| 8 | Docstring/behaviour drift (tile sizes, "whole-image" text) | tile_index.py, web.py | Misleading maintenance |

Defect 1 should be fixed regardless of which model is chosen, and checked on the benchmark (Phase 0) with portrait photos included.

### 1.4 What already exists and should be reused [code]

- `object_encoder.py`: DINOv2-with-registers ViT-B/14 CLS descriptor, 224x224, OpenVINO or torch backend; weights `dinov2_vitb14_reg4_pretrain.pth` (346 MB) are already in `~/.cache/torch/hub/checkpoints` [measured]. `embed_regions` reuses the same 14-tile grid, so it has the same alignment problem as CLIP tiles (but with a far better instance-level metric).
- `object_index.py` + table `object_region_embeddings` (+ `db.iter_object_regions`): library index builder and storage.
- `object_matcher.py`: XFeat + LighterGlue (torch/ONNX) or OpenCV SIFT, features cached per image, MAGSAC homography-or-fundamental inlier counting, `is_same_object(min_inliers=20, ratio=0.3)`. Env vars still named `MEDIA_PLACE_MATCHER*`.
- `gpu_search.py`: resident VRAM matvec/gemm for the worker or XPU host.
- Installed in `.venv` [measured]: torch 2.13 (cu130), timm 1.0.28, open_clip 3.3.0, opencv 5.0.0 (SIFT, ORB present; **AKAZE not present in this build**, `xfeatures2d` absent: needs `opencv-contrib-python-headless`), onnxruntime, ultralytics (AGPL), scipy, skimage. **Not installed:** faiss, hnswlib, kornia, transformers, lightglue, romatch, sam2.

---

## 2. Candidate options

Common assumptions for estimates: 100k images (about 1.4M CLIP tiles for scale), 4 CPU cores, fp16 storage unless stated, RAM on this box is only **7 GB total** so resident matrices above ~2-3 GB must be memory-mapped, quantised (int8/PQ) or served by the worker. "Worker" = remote GPU (bigboy / Arc B70 / CUDA); GPU timings are order-of-magnitude estimates, not measured here.

Measured here (heavily loaded box, treat as upper bound) [measured]:
- DINOv2 ViT-B/14-reg, 224 input, batch 8: **0.81 s/img** forward on CPU; at 448: **3.6 s/img**. (Expect ViT-S roughly 3-4x faster, ~0.2 s at 224.)
- OpenCV SIFT 2000 kp at 1024x768: ~0.19 s (on a blurred noise image; real images are in the same range, 0.1-0.3 s). ORB 2000 kp: ~0.01 s.
- CLIP ViT-B/32, 14 tiles: 3.4 s (random weights, contention; real number likely ~1 s). So the *current* CLIP tile index already costs about as much per image as a ViT-S DINOv2 pass would at one scale.

Rough library build cost: CPU-only DINOv2-S/14 at 224 with ~14 crops/img => ~3 s/img => 100k images = 83 h on 4 cores. This is why **a GPU worker is necessary for any multi-crop DINOv2 index**; for a single 448-px image forward with dense tokens (one pass instead of 14 crops; see option R2), ViT-S/14 is ~1 s/img on CPU => 28 h/4 cores... still worker work. On a GPU (RTX-class, fp16, batch 64) ViT-S/14 at 448 is hundreds of img/s: minutes to an hour for 100k, dominated by JPEG decode.

### 2.1 Region / instance retrieval (R)

**R-A. DINOv2 global descriptor on tiles (what ObjectEncoder does today, better tiling)**
- How: crop -> 224 (aspect-preserving pad) -> CLS (or GeM of patch tokens) -> L2. Add more tile scales (e.g. 1/2, 1/3, 1/4 of the image side at 50% overlap: 1+9+25+49 windows) and a multi-scale query (box at 1x, 0.7x, 1.4x).
- Accuracy: self-supervised features are much stronger than CLIP for same-instance; DINOv2 off-the-shelf global features are competitive with older trained retrieval nets on Revisited Oxford/Paris ([memory]: ROxf-Medium mAP in the 40s-50s and RParis-M in the 70s for ViT-B..g depending on pooling; trained SOTA with re-ranking reaches 80+/90+). It is still partly semantic: expect same-category-with-similar-appearance neighbours above the true instance for textureless objects.
- Index: 384 (S) / 768 (B) dims x number of windows. With 49+ windows: S: 49x384x2 B = 38 KB/img => 3.8 GB at 100k (needs PQ/int8 to fit RAM: 1 KB per 384-d int8 => 1.9 GB). B doubles.
- Query: brute-force matmul 4.9M x 384 fp16 ~ 3.7 GB read => ~0.5-1.5 s on 4 cores (memory-bound); with FAISS IVF-PQ or HNSW: 10-50 ms.
- Build: 50 crops/img means GPU only. ViT-S fp16 on a mid GPU: about 1-2k crops/s => 100k imgs x 50 = 5M crops ~ 1 h. CPU: infeasible.
- Deps/licence: DINOv2 code+weights Apache-2.0, ungated on HF (`facebook/dinov2-small`, `-base`, `-with-registers-small` all `gated=False`, apache-2.0) [verified]; torch hub download needs no login.
- Robustness: scale yes only if a window exists near the right scale; rotation weak (ViT trained upright, in-plane 90 degrees hurts); perspective moderate; partial occlusion moderate.
- Upside: simple, drop-in vector search, reuses ObjectEncoder. Downside: tiling dilemma again (alignment/scale), large index, no localisation proof.

**R-B. DINOv2 dense patch tokens, pooled at query time (window-agnostic) and late-interaction matching**
- How: one forward per image at ~448 px long side (32x32 patches ViT-S/14; or 518 -> 37x37) storing patch tokens after PCA to 64-128 d. Query: the same encoder on the (aspect-preserving) crop yields q tokens (e.g. 8x8=64 tokens for a 112 px crop). Score for an image = sum or mean over query tokens of max cosine to the image's tokens (ColBERT "MaxSim"), optionally with a spatial-consistency term (tokens that match should have the same relative layout: Hough/offset voting over patch offsets).
- Why it fixes the tile problem: the match happens at patch level, so the object's position and extent in the candidate are irrelevant; partial visibility only reduces the score proportionally; localisation falls out (the voting peak = the box).
- Scale: patches are 14 px at a fixed input resolution, so scale invariance comes from a **query pyramid** (query re-encoded at x0.5, 0.7, 1, 1.4, 2) and/or an index pyramid (two resolutions). Cost 3-5x on the query side, trivial relative to the scan.
- Index size per image: 1024 tokens x 64 d x 1 B (int8) = 64 KB => **6.4 GB** at 100k (memmap on disk; fine for disk, not RAM). 16x16 pooled tokens (224 grid, or 2x2-average of the 32x32) x 64 d int8 = 16 KB => 1.6 GB: fits RAM, loses small-object detail.
- Query latency: full MaxSim brute force: Q tokens (64) x 102M db tokens x 64 d = 420 GFLOP: too slow for CPU (~5-10 s at best with int8 AVX, memory-bound 6.4 GB). Two-stage is mandatory: prefilter (R-A pooled windows or whole-image DINO) to 1-3k candidates, then MaxSim on the candidates only (3k x 1024 x 64 x 64 x 2 = 25 GFLOP => <0.5 s plus loading 200 MB from a memmap). On a GPU worker with resident int8/fp16 tokens, full brute force is about 0.1-0.3 s.
- Accuracy: strong for large and medium objects and textured items; patches of ViT-S at 448 can resolve a ~40-px object; very small things (bracelet at 3% of a 12 MP frame = <60 px at 2000 px, ~25 px at 448) are at the limit. Mitigation: index a second 896-px pass only on the worker, or crop-based re-indexing of candidates on demand (re-encode the top-200 at high res at query time, ~1 s on GPU, ~40 s on CPU, so GPU only).
- Deps: torch, timm/hub DINOv2 only (already installed). Licence Apache-2.0. DINOv3 patch features are reported to be better for dense tasks, but DINOv3 weights are gated (`gated: manual`, "DINOv3 License") [verified], need an access form and emailed URLs, so not usable "without login"; the repo's own `object_encoder.py` docstring already says to avoid it. Stay on DINOv2 (optionally the `-with-registers` variants, which have cleaner patch maps).
- Robustness: scale via pyramid, partial occlusion good, perspective moderate, rotation weak (add 90/270 rotated query variants: 3 more query passes).
- Downside: engineering (variable-size token store, memmap layout, voting), index size.

**R-C. SigLIP / CLIP with better region handling (box-prompted crops, more tiles)**
- How: keep language-image embeddings but fix preprocessing (no center-crop: letterbox), add scales, optionally mask/blur the background around the query box, use SigLIP (`ViT-B-16-SigLIP-256` webli in open_clip; SigLIP2 `timm/ViT-B-16-SigLIP2-256` Apache-2.0 ungated [verified]).
- Accuracy: better localisation of category/attribute ("gold bracelet with green stones") but still semantic. For the user's goal this is a modest improvement at best; useful as a **semantic filter** layered over instance matches (e.g. restrict to images that also contain "a bracelet").
- Index: same as the current (28 KB/img at 14 tiles; 50 tiles = 100 KB/img in fp32).
- Verdict: do not pursue as the primary method. Cheap quick-fix experiments (letterbox, exif_transpose) belong in the baseline.

**R-D. Classical local features + bag of visual words / VLAD / ASMK + RANSAC re-rank**
- How: SIFT (or RootSIFT = L1-normalise then sqrt, +10% mAP typical [memory]) up to ~1000 keypoints/img; train a 64k-word codebook (k-means on a sample); inverted index with TF-IDF (BoVW) or ASMK (aggregated selective match kernel, Tolias et al. 2013) for ~+10 mAP; query = descriptors inside the box only (a natural fit for regions, unlike global descriptors). Verify the top 100-200 with RANSAC (cv2 USAC_MAGSAC homography; <1 ms/pair when keypoints are cached).
- Accuracy: the classical gold standard for **rigid, textured** instances (landmarks, posters, packaging, labelled objects, the same wall). Very high precision after RANSAC. Fails on texture-less or deformable/transparent/reflective objects (a plain gold bracelet, a smooth couch, a flower): too few repeatable keypoints.
- Index size: 1000 kp x (128 B desc + 8-16 B geometry) => ~140 KB/img => 14 GB at 100k raw; the *searchable* inverted file only stores word ids (2-4 B per kp => 4 MB/1000 imgs; ~0.4 GB at 100k), but re-ranking needs the keypoint geometry (x, y, scale, orientation: ~8-16 B each => 1-1.6 GB) and, for matching, descriptors recomputed or hashed. Practical: store keypoint coords + word ids (+ 1-byte residual signature), and verify with **geometry from word correspondences** (spatial verification as in Philbin et al. 2007), without raw descriptors.
- Query: inverted-file voting over only the query's 50-500 keypoints: 20-150 ms at 100k; RANSAC rerank 100 x ~1 ms.
- Build: SIFT 0.1-0.3 s/img on one core => 100k imgs ~1.7-2 h on 4 cores [measured, loaded]. ORB is 20x cheaper but much weaker (binary, no scale for BoVW). CPU-feasible, no worker needed. Codebook training: minutes.
- Deps: `cv2.SIFT` is in the installed OpenCV 5.0 [measured] (SIFT patent expired 2020; BSD-3 OpenCV). FAISS (`faiss-cpu` 1.15.1 MIT [verified]) for k-means and word assignment. AKAZE is **not** in this build; needs `opencv-contrib-python-headless` 5.0.0.93 [verified, Apache-2.0] and conflicts with `opencv-python` if both installed (pick one).
- Robustness: scale/rotation invariant by construction, moderate perspective (affine-ish), strong partial occlusion. The best of all options on clutter and small query box *if* the object has texture.
- Upside: precise, explainable (inlier count, homography gives the exact quad = localisation), proven. Downside: texture-less objects, repeated structures (ratio test / visual-word burstiness), custom engineering (codebook, inverted file).

**R-E. Learned local features (SuperPoint, DISK, ALIKED, XFeat) + matchers (LightGlue, LoFTR, DKM/RoMa) as the re-ranker**
- How: same cached-keypoint pattern as `ObjectMatcher` (already implemented with XFeat+LighterGlue and SIFT). Learned detectors/descriptors are much better than SIFT under viewpoint and lighting change; learned matchers (LightGlue) beat ratio-test NN; dense matchers (LoFTR, DKM, RoMa) work on weak texture but cost far more.
- Licences [verified, LightGlue README]: LightGlue code and weights Apache-2.0; DISK Apache-2.0; ALIKED BSD-3; **SuperPoint weights/inference file: restrictive licence** (MagicLeap; fine for personal non-commercial but not "permissive"); RoMa code MIT (`romatch`, default 560->864 px; Tiny RoMa on XFeat exists) [verified README]; XFeat Apache-2.0 (per `object_matcher.py` docstring); LoFTR Apache-2.0 [memory]. `kornia` 0.9.0 (Apache-2.0, py>=3.11) provides DISK, LightGlue, LoFTR, with weight download [verified on PyPI; weight hosting not tested]. `lightglue` has no PyPI JSON hit [verified failure]; the README says clone + `pip install -e .` (use `uv pip install git+https://github.com/cvg/LightGlue`), and Transformers also ships LightGlue (`transformers` 5.20.2).
- Speed [verified README + memory]: LightGlue on RTX 3080 ~150 FPS pairs at 1024 kp, ~20 FPS at 512 kp on an i7-10700K CPU. Extraction (ALIKED/DISK) on CPU is ~0.3-1 s/img, so index extraction wants the GPU worker; matching top-50 cached pairs on CPU at 512-1024 kp is 2-5 s: tight. XFeat is the CPU-friendly option (~real-time per the repo docstring).
- Cached features: 1024 kp x (64-128 d fp16 + 4 B xy) => 140-260 KB/img => 14-26 GB at 100k (disk; only candidate pairs are loaded).
- Accuracy: top re-ranker (large gains over SIFT RANSAC at wide baselines), still needs texture; RoMa/DKM handle low texture (uniform wall-ish, but not truly featureless).
- Use: **re-rank stage only**, never as first stage.

**R-F. Segmentation first (SAM / SAM2 / MobileSAM) then embed the exact mask**
- How: user box -> SAM box-prompt mask -> masked crop (background removed / mean-filled) -> descriptor. At index time one would need automatic mask generation per image (SAM AMG: 50-100 masks/img, ~1-5 s/img on GPU). `sam2` 1.1.0 Apache-2.0 on PyPI, `facebook/sam2.1-hiera-tiny` Apache-2.0 ungated, 39 M params [verified].
- Value: **query side** is cheap and strongly beneficial (removes the background that makes tight rectangles contain 30-60% non-object pixels; the drawn rectangle around a bracelet is mostly wrist). Index side is expensive and is a build cost 10-100x any other option; mask proposals also fragment objects.
- Verdict: use SAM only for the query mask (one call per search, 0.1 s on GPU, ~1-3 s on CPU with MobileSAM/SAM2-tiny), masking non-object pixels in the query crop before the DINOv2 encoding; do not segment the library. For R-B, the mask also selects which **query tokens** participate (drop background tokens), which is a substantial accuracy gain.

**R-G. Template matching (normalised cross-correlation) on pyramids**
- Works for rigid, same-scale, same-orientation copies (screenshots, scans, a logo). Not an instance retriever for 3-D scenes. Included under P for the repeating-pattern case.

### 2.2 Pattern / texture-instance retrieval (P)

First, split the problem. The user's example, "a specific spot on a wall / wallpaper / fabric / floor", covers two quite different cases:

- **P-periodic**: repeating textures (wallpaper print, fabric weave, tiles, parquet). Identity is the *motif/period and colour/arrangement*; position within the pattern is irrelevant; scale and rotation vary between photos.
- **P-aperiodic**: a unique region (stain, crack pattern, brick arrangement, a particular painting on a wall). Identity is the spatial arrangement; this is instance retrieval of a textured planar region and R-D/R-E (keypoints + homography) is by far the strongest.

A cheap autocorrelation periodicity test on the query crop decides the route (see P-F).

**P-A. DINOv2 patch tokens + MaxSim (same machinery as R-B), texture-restricted**
- Texture crops produce a *set* of similar tokens; MaxSim against a candidate's tokens finds the same texture anywhere in the photo, at any extent (even if it occupies only 3% of the photo), without defining windows. Query pyramid for scale; rotated queries (0/90/180/270) for orientation, since ViT tokens are not rotation-invariant.
- Compared with global features of tiles this fixes "pattern is a small part of the tile". Compared with LBP it adds learned invariances to illumination and some perspective and distinguishes motif-level detail.
- Concern: DINO tokens encode semantics too (a "sofa" token vs "wall" token), which is helpful for finding the same material but may match other wallpapers with similar content-level appearance. Include a **Gram / second-order statistic** variant (P-C) as an alternative scoring.
- Index/latency identical to R-B (shared index: huge saving, one dense-token index serves R and P).
- Accuracy: expected best of the learned options for periodic textures at moderate scale; weak at the extremes (very fine weave under 14 px/patch at 448 input).

**P-B. Classical, done properly (rotation/scale-invariant LBP + Gabor/wavelet + colour moments with a proper metric)**
- How: multi-scale (3 scales x radii 1,2,3), rotation-invariant LBP (`riu2`, 10 bins/scale) or LBP-HF (Fourier features of LBP histograms), steerable-pyramid or wavelet-packet energy (log, per scale, sorted over orientation for rotation invariance), colour moments in Lab (mean/std/skew) after grey-world normalisation; sqrt (Hellinger) per block, per-block L2 and block weights *learned* (metric learning on a handful of labelled pairs; e.g. LDA/whitening fitted on the library tiles).
- Descriptors computed on **windows at the query's native scale**, for many windows per image: multi-scale sliding windows (dense stride 50%, scales 1/8..1/2 of side) => 100-300 windows x ~100 d. 100 windows x 100 d x 1 B = 10 KB/img => 1 GB; CPU build ~20-50 ms/img if windows are described via integral histograms (LBP code map computed once; histograms per window from a 2-D integral image per bin).
- Accuracy: decent for stochastic fine textures (fabric, grain, stucco), poor on designed motifs (wallpaper pattern with distinct print). Better than the current descriptor by a wide margin, mostly from fixing scale, windows, metric. Does not separate two different prints of similar texture energy.
- Cost: zero GPU, zero new deps. Good fast prefilter and a clean baseline.
- Licence: n/a (numpy/OpenCV/skimage).

**P-C. Deep texture statistics: Gram matrices / Fisher vectors / VLAD of dense features, bag of textons**
- How: take dense local features (VGG/ResNet conv layer, or DINOv2 tokens) of the crop and of each candidate window; encode as Fisher vector (FV-CNN, Cimpoi et al., CVPR 2015, https://arxiv.org/abs/1411.6836, which gave large gains on DTD/FMD over hand-crafted textures), VLAD, or Gram matrix (Gatys et al. 2015, https://arxiv.org/abs/1505.07376). Orderless pooling gives translation invariance, partial scale/rotation invariance.
- Accuracy: excellent for "same material class" (DTD: 47 classes). For *instance* of a specific printed pattern, an orderless descriptor loses arrangement, but print motifs are mostly local, so it works moderately. Gram vectors are large: C(C+1)/2 for C=64 => 2080 d (dimension reduce by PCA).
- Index: per window 128-d FV/VLAD => same as P-B but with CNN cost (window pooling from a stored feature map is cheap: store a 32x32x(64-128) token map as in R-B, and compute **window statistics at query time from the same stored tokens**). So P-C is mostly a *scoring variant* on the R-B token store; cost is incremental.
- Deps: torch/timm only. VGG-16 weights (BSD-style via torchvision; fine).

**P-D. Template matching / FFT NCC on multi-scale pyramids (exact repeats)**
- How: `cv2.matchTemplate(TM_CCOEFF_NORMED)` or FFT-based NCC of the query crop against each candidate's downscaled grayscale (e.g. 640 px long side), over a scale set {0.5 .. 2} in steps of ~1.2 (7 scales) and rotations (0, 90, 180, 270, or ±15 degrees if needed). Peak NCC > ~0.6-0.7 with a sharp peak (peak-to-sidelobe ratio) is a hit.
- Accuracy: near-perfect precision for **pixel-level same patterns** (same wallpaper roll, same print on fabric photographed head-on). Collapses under perspective, strong lighting change (NCC is invariant to affine intensity change globally but not to shadows), or when the query contains a non-repeating part; very sensitive to scale step (needs <10% steps).
- Cost: for a 100x100 template on a 640x480 image, `matchTemplate` takes ~3-10 ms per scale on one core (OpenCV picks DFT when large) => 7 scales x 4 rotations x 5 ms = 140 ms/img/core. Not for a 100k scan (14 000 s/core => ~1 h/4 cores for a *single* query), fine for **re-ranking the top 200-500** from a prefilter (~15-30 s on one core, 4-8 s on 4 cores; faster with downscaled 320 px). On GPU torch FFT-conv: ~1 ms per image-scale => the full library scan in ~10 minutes: only a worker thing.
- Deps: OpenCV only.

**P-E. Lattice / periodicity detection for wallpaper tiles**
- How: autocorrelation (FFT) of the grayscale query; the lattice vectors give period (px) and orientation; wallpaper group could be estimated but is not needed. Use for three things: (1) decide periodic vs aperiodic route; (2) **normalise scale**: rescale the query crop to canonical period (e.g. 32 px) and, at the index side, estimate the dominant period per window (autocorr peaks) and store `(period, angle)` with each window descriptor, so matching compares **scale- and rotation-normalised unit cells** (resample the window to canonical period/orientation before computing the descriptor); (3) provide a clean **unit-cell template** for NCC verification (P-D) and the lattice makes template matching rotation-aware.
- Accuracy: This is the only approach that gives true scale/rotation invariance for periodic patterns, which is what a wall-paper-finding query needs. Fails for non-periodic or weakly periodic textures (then fall back to P-aperiodic/R-D) and under strong perspective (period varies across the image; compute locally on windows).
- Cost: FFT autocorr on 256x256: ~3 ms; per-window period estimation at index time 100-300 windows x 3 ms => ~0.5 s/img/core: that is 3.5 core-days at 100k, so restrict to a coarse grid (16-32 windows, 15-50 ms/img) or do it on the worker. Alternatively do the period estimation at query time only on the **prefiltered candidate windows** (cheap).
- Deps: numpy FFT only.

**P-F. Hybrid two-stage for P (recommended structure)**
1. Query: crop -> periodicity test (autocorr) -> route.
2. Stage 1: prefilter with P-B windows (CPU, resident, <200 ms) and/or pooled DINOv2 windows to get 1-2k candidates.
3. Stage 2 (periodic): normalised-unit-cell NCC (P-D/P-E) + DINOv2 token MaxSim; stage 2 (aperiodic): RootSIFT/XFeat + RANSAC (R-D/R-E).
4. Fuse scores by rank (RRF) with a calibrated threshold, show matched box.

### 2.3 Cross-cutting: two-stage hybrids and ANN indexing

- **Why two stages:** prefilters are cheap, high-recall, low-precision (pooled embeddings); re-rankers are expensive, high-precision, need per-pair compute (RANSAC, MaxSim, NCC). With a 1-3k shortlist all re-rankers fit in 0.5-2 s on 4 cores if their per-image features are cached.
- **Prefilter vectors:** pooled DINOv2 windows (R-A) are ideal, plus a whole-image DINO/CLIP vector for the "full-frame" case.
- **ANN options** (at ~1.4M-5M vectors x 384-512 d):
  - *Plain numpy matmul in chunks over a resident fp16/int8 matrix:* zero deps, exact. 1.4M x 512 fp32 = 2.9 GB (fp16 1.4 GB, int8 0.7 GB); a mat-vec is memory-bound: ~0.15-0.5 s per query on 4 cores if resident. The biggest wins here are not ANN but *not re-reading SQLite per query*, using per-file `np.maximum.reduceat` instead of Python loops, and keeping a resident int8/fp16 copy. numpy cannot matmul fp16 fast on CPU: convert chunks to fp32, or use torch bf16/`torch.mm` on int8-dequant.
  - *FAISS CPU* (`faiss-cpu` 1.15.1, MIT, py>=3.10 wheels [verified]): `IndexIVFPQ` / `IndexHNSWFlat` / `IndexFlatIP` with fp16 SQ (`IndexScalarQuantizer`). IVF-PQ (nlist 4096, m=64, 8 bit) = 64 B/vec => 5M vectors 320 MB, ~5-20 ms per query at nprobe 16, recall@100 ~0.9 for 384-d after rerank [memory]. Needs a train pass (100k-1M samples, minutes). Best choice once windows exceed ~3M vectors or RAM is tight.
  - *hnswlib* 0.8.0 [verified exists on PyPI; sdist only per empty licence/py metadata, expect a C++ build step]: high recall, 1-2 ms, but graph memory (M=16: +128 B/vec) and no compression: 5M x 384 fp32 = 7.7 GB: too big for this box.
  - *Per-file max reduction:* ANN returns tile ids; map to file ids with a `tile_to_file` int32 array; take top 5x candidates.
- **On the worker:** `gpu_search.matvec` keeps a matrix resident; a 5M x 384 fp16 matrix is 3.8 GB VRAM, a mat-vec takes ~1-3 ms. This is the simplest high-performance option for the prefilter if the worker is always up. A CPU path must still exist for when it is down (repo rule: documented fallback).

---

## 3. Comparison table

Index size/query at 100k images. "Lat." = query latency on the 4-core CPU without GPU unless noted. Accuracy rating is expected quality for the stated user task (1 poor, 5 excellent), my judgement from literature, not measured on the benchmark.

| Option | Task | Index / img | Build (100k) | Query lat. | Rot. | Scale | Persp. | Occl. | R acc. | P acc. | Licence / deps | Main risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Current CLIP B/32 tiles | R | 28 KB | done | 5-60 s (SQLite scan) | - | - | - | - | 1-2 | 1 | open_clip | semantic, misaligned |
| Current LBP+HSV+Gabor | P | 0.6 KB x14 | CPU, but ships files to worker | 3-30 s | no | no | no | no | - | 1 | numpy/cv2 | scale mismatch, colour-dominated |
| R-A DINOv2 pooled windows (S/14, 50 windows) | R,P prefilter | 38 KB fp16 (19 KB int8) | GPU ~1 h; CPU infeasible | 50 ms (IVF-PQ) to 1 s (flat) | weak | window dep. | mod. | mod. | 3 | 2-3 | Apache-2.0 HF ungated | tiling dilemma |
| R-B DINOv2 dense tokens + MaxSim (rerank 2k) | R,P | 16-64 KB (int8, 64 d) | GPU 20-60 min; CPU ~28 h | 0.3-1 s with prefilter | via 4 rotated queries | via pyramid | mod. | good | 4 | 4 | Apache-2.0; torch | index size, small objects |
| R-C SigLIP/CLIP improved tiles | R | 28-100 KB | GPU | 0.3-1.5 s | weak | tile dep. | - | - | 2 | 1 | open_clip, Apache | still semantic |
| R-D SIFT/RootSIFT BoVW/ASMK + RANSAC | R, P-aperiodic | ~14 KB words+geom (1 GB total) + descriptors for verify | CPU ~2 h | 0.1-0.5 s | yes | yes | affine-ish | good | 4 (textured) / 1 (plain) | 3 (aperiodic) / 1 (periodic) | OpenCV SIFT, faiss | smooth/deformable objects |
| R-E ALIKED/DISK/XFeat + LightGlue re-rank | R re-rank | 140-260 KB cached | GPU ~1-2 h | 2-5 s for 50 pairs CPU | yes | yes | good | good | 5 (textured) | 3 | LG Apache; DISK Apache; ALIKED BSD; SuperPoint restrictive | GPU for extraction |
| RoMa/DKM/LoFTR dense re-rank | R re-rank | none (pairwise) | - | 0.3 s/pair GPU; 5-20 s/pair CPU | good | good | good | good | 5 | 4 | MIT/Apache | GPU only, per-pair |
| R-F SAM/SAM2 on query only | R,P | none | - | +0.1 s GPU / 1-3 s CPU | n/a | n/a | n/a | n/a | +1 acc level | +1 | Apache-2.0, ungated | box-to-mask errors |
| SAM on library | R | masks x features | days | - | - | - | - | - | 4 | - | - | 10-100x build cost |
| P-B classical done right (riu LBP, wavelets, Lab moments, Hellinger, windows) | P | ~10 KB | CPU ~1 h | 0.1-0.3 s | yes | multi-scale | no | mod. | - | 3 (stochastic) / 2 (motif) | none | no motif identity |
| P-C Gram/FV/VLAD of dense feats | P | 128-d x windows (derived from token store) | cheap on token store | 0.2-1 s | partial | partial | partial | mod. | - | 3-4 | torch/timm | arrangement lost |
| P-D FFT NCC multi-scale (rerank 300) | P | none | none | 5-10 s CPU rerank (1-2 s at 320 px, fewer scales) | with rotated queries | 7 scales | no | poor | - | 5 (exact repeats) / 1 | OpenCV | brittle to perspective/lighting |
| P-E autocorrelation lattice normalisation | P | +8 B/window | CPU 0.05-0.5 s/img | negligible | yes | yes | no | - | - | 4 (periodic) | numpy | aperiodic patterns |
| Hybrid H (prefilter + geometric/token rerank) | R,P | R-A + feat cache | worker 1-3 h | 0.5-2 s | yes | yes | good | good | 4-5 | 4 | as above | complexity |

---

## 4. Recommended shortlist (experiments)

All five reuse one benchmark/harness (Phase 2): recall@k, MRR, mAP per task. Install nothing until the orchestrator approves; commands below use `uv` as per repo policy.

**Prerequisite for every experiment (not a separate arm): fix EXIF orientation** (`ImageOps.exif_transpose` on every decode path used for query and index; ensure the pattern worker decode path and the PIL path agree). Include portrait (orientation 6/8) photos in the benchmark and report with/without the fix; this alone may change the baseline noticeably.

### E1 (R, primary): DINOv2 windows prefilter + RootSIFT/XFeat geometric re-rank (the hybrid H for R)
- Prefilter: `facebook/dinov2-with-registers-small` (HF, Apache-2.0, ungated, 22 M params) or `dinov2_vits14_reg` from torch hub; **aspect-preserving letterbox at 224**, windows at 1/1 (whole), 1/2, 1/3, 1/4 of the side with 50% overlap (1+9+25+49 = 84 is too many; start with 1/2, 1/3, 1/4 => ~34), L2-normalised CLS, stored as fp16. Query: multi-scale (box 1x, 0.75x, 1.33x) max. Search with a resident matrix; `faiss-cpu` only if needed.
- Re-rank top 200: SIFT (`cv2.SIFT.create`, installed) with **RootSIFT**, keypoints restricted to the query box, ratio 0.8 + mutual check, `cv2.findHomography(USAC_MAGSAC)`; features cached per image (reuse `ObjectMatcher` sift/xfeat backend and `is_same_object`). Final score = inliers (with DINO score as tie-break), min-inlier threshold to remove same-category hits. Localise via the homography quad.
- Packages: nothing new on CPU (torch, timm, opencv present); `uv pip install transformers` only if using the HF checkpoint (timm alternative: `timm/vit_small_patch14_reg4_dinov2.lvd142m`); optional `uv pip install faiss-cpu` (MIT, verified 1.15.1); optional XFeat via `git+https://github.com/verlab/accelerated_features` (as in `object_matcher.py` runbook).
- Build: window embedding on the GPU worker; SIFT/RootSIFT on the CPU box (100k x 0.2 s / 4 cores ~ 1.5-2 h).
- Expected: large improvement for textured/rigid instances; plain smooth objects still weak (the DINO stage carries them).

### E2 (R + P shared, dense): DINOv2 dense-token store with query-pyramid MaxSim and offset voting
- Index (worker): ViT-S/14-reg (or `vit_small_patch14_dinov2.lvd142m` via timm), long side 448 (32x32 tokens), PCA to 64-d fitted on ~50k sampled tokens, int8 quantise, memmap file `tokens.u8` + per-image offsets. ~64 KB/img (6.4 GB on disk). Optional 16x16 pooled variant (1.6 GB) for the RAM-resident prefilter.
- Query: crop (optionally SAM2-tiny mask to select tokens, `facebook/sam2.1-hiera-tiny`) at scales {0.5, 0.7, 1, 1.4, 2} x rotations {0, 90, 180, 270}; MaxSim vs a 2k-candidate shortlist from E1's prefilter; add offset-voting spatial consistency; return best box.
- Packages: nothing new for the core (torch/timm installed); `sam2` (PyPI 1.1.0 Apache-2.0) only for the optional mask.
- Expected: strongest for small/partial objects and for texture crops; and gives P almost for free.

### E3 (P, classical-but-correct baseline + prefilter): multi-scale rotation-invariant texture windows with a learned metric
- Replace the 147-d descriptor: grey-world + Lab, `skimage.feature.local_binary_pattern(method='uniform')` at (P,R) = (8,1), (16,2), (24,3) (riu2 histograms, 10+18+26 bins), wavelet/steerable energy (log, sorted over orientation), Lab moments; Hellinger per block; **windows at native query scale** (descriptor computed from the full-resolution crop, not 128-px resize), sliding windows with 50% overlap at 1/8, 1/6, 1/4 of the short side; whiten (PCA-whitening fit on library windows) so cosine is meaningful; optional pair-based weight fit.
- Query: the crop is described at the native scale and at 2 down-scalings; best over scales.
- Packages: none new (numpy, scikit-image, OpenCV all installed).
- Purpose: a fast CPU-only floor and the prefilter for E4; also tells us how much of the failure was bugs vs. descriptor class.

### E4 (P, periodic + aperiodic): periodicity-aware unit-cell NCC verification + RootSIFT RANSAC for non-periodic
- Query: autocorrelation of the crop -> period/orientation or "aperiodic". Periodic: rescale to canonical period, shortlist candidates via E3 windows and E2 tokens, verify with multi-scale (7 scales, steps 1.2) rotation-aware `cv2.matchTemplate(TM_CCOEFF_NORMED)` on 320-640 px downscaled candidates with peak-to-sidelobe check. Aperiodic: E1's SIFT/MAGSAC re-rank.
- Packages: none new (OpenCV, numpy FFT). 
- Expected: near-perfect precision for exact repeats, graceful fallthrough for non-periodic spots.

### E5 (hybrid, shared): single two-stage service
- A common `prefilter` (E1 windows + E3 windows) -> `rerank` plug-in interface (`sift_ransac`, `maxsim`, `ncc`) with a score-fusion stage (reciprocal rank fusion + calibrated threshold). Compare the *combined* system rather than isolated rerankers; include latency budget accounting (<2 s on 4 cores).

Priority if only three can be built: **E1, E2, E4** (E3 as the cheap baseline integrated into E4).

Higher-end follow-up (only if E1-E4 miss target on textured-but-wide-baseline cases): swap SIFT for ALIKED/DISK + LightGlue (`kornia` 0.9.0 Apache-2.0, or `git+https://github.com/cvg/LightGlue`) as the re-ranker on the GPU worker; try RoMa (`romatch`, MIT) for texture-poor pairs. Avoid SuperPoint (restrictive weights licence) and DINOv3 (gated, custom licence).

---

## 5. Integration notes (for later phases)

- Storage: replace per-query SQLite scans with an on-disk `.npy`/memmap index built from the DB (versioned by model + algo tag; reject mismatches). Keep SQLite as the source of truth for rebuild; add `model`/`algo` check at query.
- Worker: embedding and token extraction go to the worker; **do not ship full-size originals** (downscale to 448-896 px and JPEG q85 first; the existing region-crop path already sends "cropped+downscaled JPEG"). Pattern classical code needs no worker at all.
- Fallback honesty: if no tokens/keypoints are indexed for a file, say so in the response (`coverage`), not a silent whole-image fallback.
- Results should carry `bbox` (best window / homography quad) so the UI can draw it and the harness can score localisation.
- Memory: this web host has 7 GB RAM; keep resident structures below ~2 GB (int8/PQ) and memmap the rest.

## 6. Sources / references

- DINOv2: https://arxiv.org/abs/2304.07193 ; code https://github.com/facebookresearch/dinov2 ; HF `facebook/dinov2-small|base|large|with-registers-*` (apache-2.0, `gated: False`, verified via HF API).
- DINOv3 (gated, "DINOv3 License", manual access, verified via HF API and README): https://github.com/facebookresearch/dinov3
- Registers: https://arxiv.org/abs/2309.16588
- LightGlue (Apache-2.0; SuperPoint restrictive; DISK Apache-2.0; ALIKED BSD-3; speed numbers): https://github.com/cvg/LightGlue , paper https://arxiv.org/abs/2306.13643
- RoMa (MIT, `romatch`): https://github.com/Parskatt/RoMa ; DKM https://github.com/Parskatt/DKM ; LoFTR https://github.com/zju3dv/LoFTR
- XFeat: https://github.com/verlab/accelerated_features , https://arxiv.org/abs/2404.19174
- SuperPoint: https://arxiv.org/abs/1712.07629 ; DISK https://arxiv.org/abs/2006.13566 ; ALIKED https://arxiv.org/abs/2304.03608
- Revisited Oxford/Paris benchmark: https://arxiv.org/abs/1803.11285 ; Google Landmarks v2: https://arxiv.org/abs/2004.01804
- RootSIFT: Arandjelovic and Zisserman, CVPR 2012; ASMK: Tolias, Avrithis, Jegou, "To aggregate or not to aggregate", ICCV 2013 / IJCV 2016; spatial verification: Philbin et al., CVPR 2007; DELF/HOW/SuperGlobal for modern re-ranking: https://arxiv.org/abs/2308.06954 (SuperGlobal)
- Textures: FV-CNN https://arxiv.org/abs/1411.6836 ; DTD https://arxiv.org/abs/1311.3618 ; Gram matrices https://arxiv.org/abs/1505.07376
- ColBERT late interaction (MaxSim idea): https://arxiv.org/abs/2004.12832
- SAM2: https://github.com/facebookresearch/sam2 (`sam2` 1.1.0 on PyPI, `facebook/sam2.1-hiera-tiny` apache-2.0, ungated, verified)
- SigLIP2 (timm `timm/ViT-B-16-SigLIP2-256`, apache-2.0, ungated, verified): https://arxiv.org/abs/2502.14786
- FAISS (`faiss-cpu` 1.15.1, MIT, verified on PyPI): https://github.com/facebookresearch/faiss ; hnswlib 0.8.0 (PyPI, builds from source).
- Verification caveats: literature accuracy figures are from memory (web search tool credits were exhausted); PyPI lookups for `romatch` and `lightglue` returned no JSON (install via git/`uv pip install romatch` per README should be checked before use); `transformers` 5.20.2, `kornia` 0.9.0 (needs Python >= 3.11; venv is 3.12, fine).
