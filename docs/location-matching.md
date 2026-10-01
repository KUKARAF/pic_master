# Location matching by place (people ignored)

Location "find similar" used the whole-image CLIP embedding, which is dominated by
**people**, so it matched faces instead of the place. This replaces it with a
dedicated **Visual Place Recognition** signal computed with people masked out, plus a
local-feature **geometric re-rank** for "literally the same spot".

## Pipeline

1. **Stage 1 — scene embedding** (`place_encoder.py`, default **AnyLoc** = DINOv2 patch
   tokens + VLAD). People's patch tokens are dropped (boxes from YOLO person detections)
   before aggregation, so the descriptor keys on the background/scene. Stored in
   `place_embeddings` (sentinel = "no usable place signal", won't re-queue).
2. **Stage 2 — geometric re-rank** (`place_matcher.py`, default **XFeat + LighterGlue**).
   Keypoints are extracted once per image (person-box keypoints dropped) and cached in
   `place_keypoints`; at query time the top ~30 candidates are matched against the
   location's own photos and re-ordered by RANSAC (MAGSAC, F & H, max inliers).

The location ranking (`_find_similar_files_for_location`) uses place embeddings when
the index is built, and **falls back to CLIP** (old behaviour) when it isn't — so
nothing breaks before you build the index. The swipe UI / accept-reject are unchanged.

## B70 setup (prod GPU host)

Both models run **locally on the B70** (that's where the Arc GPU is — not offloaded to
the worker). Full runbooks are in the module docstrings:
`media_manager/place_encoder.py` and `media_manager/place_matcher.py`. In short:

1. Install runtime: `openvino` (≥ **2025.3**, for the XFeat NMS fix), `optimum-intel`
   (conversion only), `onnxruntime-openvino`, `accelerated_features` (XFeat), and the
   DINOv2 backbone.
2. Convert DINOv2 to OpenVINO IR (e.g. `optimum-cli export openvino --model
   facebook/dinov2-giant --task image-feature-extraction <out>`), and place the AnyLoc
   VLAD vocabulary under `<library>/.media/place_vocab/`. Env: `MEDIA_PLACE_MODEL`,
   `MEDIA_PLACE_DEVICE=GPU`, `MEDIA_PLACE_DINOV2_OV`, `MEDIA_PLACE_FACET`,
   `MEDIA_PLACE_MATCHER`. Optional PCA whitening at `<library>/.media/place_pca.npz`.
3. Verify the graph runs on the GPU (no silent CPU fallback) — the modules log device
   placement and raise loud errors with the exact fix command when deps/weights/vocab
   are missing (no silent degrade).

Bring-up tip: set `MEDIA_PLACE_MODEL=eigenplaces` first (a plain ResNet that converts
to OpenVINO trivially) to validate the whole index→rank→swipe path end to end, then
switch to `anyloc` — the table/job/ranking are encoder-agnostic.

## Build the index

For masking to work, object detections must exist first (run tag/object reindex).
Then on **/bulk** click **📌 Index places** (or `POST /api/place-index/start`; poll
`/api/place-index/status`). It computes the scene descriptor + cached keypoints for
every image, people masked, skipping already-indexed files. Re-run after adding photos.

Until the index exists, location matching transparently uses the old CLIP ranking.
