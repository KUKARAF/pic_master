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

## Setup

Both models run **locally on the GPU host** (that's where the Arc GPU is — not offloaded
to the worker). Backend selection is **automatic** (`MEDIA_PLACE_MODEL=auto` /
`MEDIA_PLACE_MATCHER=auto` by default).

### Baseline — zero setup (just works)
Out of the box the encoder resolves to **EigenPlaces** (runs on the torch the app already
has, auto-downloads its own small weights) and the matcher to **SIFT** (OpenCV, already
present). People are ignored by **pixel-masking** the person boxes from object detection.
So: reinstall, (optionally `media place-setup` to pre-download), run object detection so
masking has boxes, then **📌 Index places** on /bulk. No OpenVINO, no vocab, no env vars.

### Upgrade — AnyLoc + XFeat (best quality, GPU, token-level people-dropping)
Install the extra deps **into the app's venv** (Python **3.11–3.13** — `onnxruntime-openvino`
has no 3.14 wheel; the prod `media-prod` venv is 3.12):

```
uv pip install --python /home/rafa/media-prod/bin/python openvino optimum-intel accelerated_features
media place-setup --anyloc    # converts DINOv2 → OpenVINO IR; prints the one remaining vocab step
```

`place-setup --anyloc` converts the DINOv2 backbone to OpenVINO IR under
`<library>/.media/dinov2_ov/`; add the AnyLoc VLAD vocabulary under
`<library>/.media/place_vocab/` (see `place_encoder.py`'s runbook for fitting it over the
served facet). Once both are present, `auto` upgrades the encoder to AnyLoc on its own —
no config change, no index rebuild semantics change (the `model` column versions it, so a
re-index picks up the better descriptor). `onnxruntime-openvino` is optional (only the
XFeat *ONNX* serving path uses it; XFeat otherwise runs via torch). Full runbooks:
`media_manager/place_encoder.py`, `media_manager/place_matcher.py`.

## Build the index

For masking to work, object detections must exist first (run tag/object reindex).
Then on **/bulk** click **📌 Index places** (or `POST /api/place-index/start`; poll
`/api/place-index/status`). It computes the scene descriptor + cached keypoints for
every image, people masked, skipping already-indexed files. Re-run after adding photos.

Until the index exists, location matching transparently uses the old CLIP ranking.
