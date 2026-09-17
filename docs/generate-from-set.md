# Generating images & video from a set

Status: **planning + Phase 1 in progress** (last updated 2026-09-17).

A "set" is a named collection of the user's own media (often a studio shoot, an
event, or one person). This doc surveys the options for producing a **new image**
and a **new video** *from* a set's members, picks a best-fit tech stack for each
medium **separately**, and lays out a phased plan. All generated output is kept
apart from the library in a `generated/` folder and marked (see
[Provenance & segregation](#provenance--segregation)).

Target hardware is an **Intel Arc Pro B70** (Battlemage, 32 GB) on Linux/Fedora;
the app already runs `torch 2.7.1+xpu` and `onnxruntime` with the OpenVINO
execution provider.

> **Scope (clarified 2026-09-17):** the goal is *generative*, not compositing.
> **Video** = *extrapolate frames BETWEEN set items* to make a **morph** (motion
> synthesized from the set's photos). **Images** = *synthesize NEW images*
> conditioned on the set. A slideshow, Ken Burns, collage, or contact sheet is
> explicitly **NOT** the deliverable — those are static composites of the
> originals. Both real features are model-based and therefore run on the **B70
> box** (no GPU/models on the dev machine, so unlike a composite they can't be
> exercised locally).

---

## TL;DR

| | Near-term engine | Best quality | Later |
|---|---|---|---|
| **Images** | face-swap (`inswapper`, reuses our InsightFace) for person-sets | identity gen — InstantID / PuLID / **Qwen-Image-Edit** via **OpenVINO GenAI**; img2img/IP-Adapter for scenes | per-set **LoRA** (best identity; Arc *training* is the weak spot) |
| **Video (morph)** | **frame interpolation** between consecutive set items — RIFE (Vulkan, runs on Arc) / FILM — assembled to mp4 (hardware-encoded on the B70) | **Wan 2.2 FLF2V** — *generated* motion between two keyframes (true in-between frames, not a warp) via Intel `llm-scaler` ComfyUI-XPU; LTX-2 for single-image animation | — |

Everything here is generative/model-based and built/verified on the **B70 box**.
The only pieces that run anywhere (and are already implemented + tested) are the
support layers: the segregated `generated/` folder, the `generated_artifacts`
registry, and provenance marking.

---

## The B70 reality (grounds every compatibility claim)

- **Hardware:** Arc Pro B70, Battlemage, **32 GB GDDR6**, 608 GB/s, 367 INT8 TOPS,
  dual media engines (AV1/HEVC/H.264 encode+decode). The 32 GB removes the VRAM
  wall that blocks 16–24 GB cards; bandwidth (~half a modern NVIDIA card) means
  it's *usable, not fast*.
- **Runtime pecking order for gen-AI (most→least reliable):**
  1. **OpenVINO GenAI / optimum-intel** — Intel-blessed, most reliable; SDXL,
     FLUX, Qwen, and the **LTX-Video** pipeline. Clean Python API to embed.
  2. **ComfyUI via Intel `llm-scaler` XPU container** — pragmatic for the node
     ecosystem (IP-Adapter/ControlNet) and for video (Wan 2.2, LTX-2).
  3. **Native `torch.xpu` + diffusers** — flexible and it's our existing stack,
     **but** an open bug (`torch-xpu-ops` #5220) makes fp16/bf16 `nn.Linear`
     return zeros/NaN on Battlemage, which **breaks SDXL** on the raw XPU path.
     Use OpenVINO for diffusion; keep XPU for the ONNX/face path only.
  4. **A1111** — fragile on XPU; avoid for a service.
  - **Faces / ONNX:** `onnxruntime` + OpenVINO EP is solid and already in use.
- **Hard caveats:** #5220 (fp16 SDXL) → prefer OpenVINO; many video sampler nodes
  need fp64 (Arc lacks it) → crashes/CPU-fallback; kernel floor is strict (`xe`
  driver, Linux 6.17+, compute-runtime + Level-Zero + GuC/HuC firmware). **All
  generation is an async job — never inline in a request.**
- Speeds below are **extrapolated** (B70 ≈ 2× B580; no B70-specific gen
  benchmarks published as of 2026-09) — validate on the card.

---

## Options surveyed

### Family 1 — Deterministic composition (no AI)
| Option | Makes | Realism | B70 | Effort |
|---|---|---|---|---|
| Grid/collage, contact sheet (Pillow) | image | perfect (real paste) | CPU, instant | Low |
| Photomosaic (target tiled from set photos) | image | real tiles (by design) | CPU | Med |
| Poster/cover composite (Pillow+OpenCV) | image | real + text | CPU | Low–Med |
| Slideshow / Ken Burns / xfade (ffmpeg) | video | perfect | **GPU encode (AV1/HEVC)** | Low |
| RIFE frame interpolation (smoothing) | video | slight artifact risk | Vulkan → runs on Arc | Med |

### Family 2 — Generative still images
| Option | Identity fidelity | Realism | B70 path | Effort |
|---|---|---|---|---|
| img2img / variation | low | high | OpenVINO / XPU | Low |
| IP-Adapter (style/subject) | moderate | high | OpenVINO / ComfyUI-XPU | Low–Med |
| ControlNet (pose/depth/edge) | n/a (structure) | high | ComfyUI-XPU | Med |
| InstantID / PuLID / PhotoMaker (few-shot) | high, no training | high | ComfyUI-XPU (32 GB fits FLUX) | Med |
| **Qwen-Image-Edit-2511** (instruction edit) | high | SOTA-open, **Apache-2.0** | **OpenVINO GenAI** (FP8 ~16 GB) | Med |

### Family 3 — Person-consistent (this app's sweet spot; sets are often one person)
| Option | Best at | B70 | Notes |
|---|---|---|---|
| **InsightFace `inswapper`** (already shipped) | literal face swap, highest likeness | **onnxruntime-OpenVINO — existing path** | needs GFPGAN/CodeFormer sharpen; non-commercial license |
| PuLID-FLUX / InstantID | person in new scenes | ComfyUI-XPU | reuses our face embeddings |
| LivePortrait | expression/motion from a driving video | XPU/OpenVINO, light | non-commercial |
| Hallo3 (MIT) | audio talking-head, long clips | XPU, heavy/experimental | phase-3 |

### Family 4 — Generative video
| Option | Makes | Realism | B70 | Time/clip |
|---|---|---|---|---|
| **LTX-2** | image→video, short clips | good, fast | **OpenVINO GenAI (embeddable)** or llm-scaler | ~1–3 min |
| **Wan 2.2** (5B fast / 14B-GGUF quality) | I2V + **FLF2V morph between 2 photos** | strongest open | Intel `llm-scaler` ComfyUI-XPU | ~5–12 min |
| SVD / AnimateDiff | I2V / motion | legacy, warpy | runs, unoptimized | — |
| CogVideoX / Mochi | — | — | **poor Arc fit / text-only — skip** | — |

### Family 5 — Per-set LoRA/DreamBooth
Highest identity ceiling ("this set = this person, anywhere"); **Arc *training* is
the weakest Battlemage story** (CUDA-centric trainers). LoRA *inference* on
XPU/OpenVINO is fine. Phase-3.

---

## Provenance & segregation

Non-negotiable design rule (per product decision): **every artifact this feature
creates is kept out of the normal library and clearly labelled.**

- **Location:** all outputs are written under `<data_root>/generated/`, never
  mixed into the user's source folders. Subfoldered by set and kind, e.g.
  `generated/<set-slug>/faceswap-<ts>.jpg`, `generated/<set-slug>/morph-<ts>.mp4`.
- **Excluded from normal indexing/browsing/search** by default — the `generated/`
  tree is skipped by the scanner so a generated image never re-enters the library
  as if it were an original. Generated items are browsed from a dedicated view (a
  `/generated` page and/or the set's own "Generated" strip).
- **Provenance marking:**
  - `origin = "ai"` — anything a diffusion/video/interpolation model synthesized.
    Labelled **"AI"** with a visible badge (modeled on the existing 🗑 trash
    badge pattern), and **embedded metadata** on the file itself: EXIF/XMP
    `Software`/`DigitalSourceType` = "AI-generated" (and C2PA content credentials
    if we add the lib later), so the mark survives download/export.
- **DB:** record each generated file with its `origin`, the source `set_id`, the
  technique/model, and generation params (prompt/seed) for reproducibility.
- **Never auto-mixed into feeds** (Random/Unloved/etc.) unless the user opts in.

This section applies to **both** plans below.

---

## Plan A — Images (synthesize a NEW image from a set)

**Goal:** synthesize a *new* image conditioned on the set — a person from a
person-set placed in a new scene, or a new image in the set's look. (Not a
collage of the originals.)

**Chosen stack (independent of the video stack), all on the B70:**
- **Near-term — face-swap:** **`inswapper`** (reuses the InsightFace models
  already in the app, runs on our onnxruntime-OpenVINO path) + a GFPGAN/CodeFormer
  restore to sharpen the 128px swap. Lowest effort, highest literal likeness for
  person-sets; puts the set's person into a target/generated image.
- **Best quality — identity generation:** an **OpenVINO GenAI** image service on
  the reliable path (SDXL / **Qwen-Image-Edit-2511**, Apache-2.0), plus a
  **ComfyUI-XPU** graph for **InstantID / PuLID / IP-Adapter + ControlNet**
  stacking. Feed 3–5 crops from the set for identity.
- **Later — per-set LoRA:** SDXL LoRA trained via IPEX/XPU (background job) for
  strong repeatable identity; inference via OpenVINO. (Arc *training* is the weak
  spot — the one place CUDA would help.)

Output → `generated/…`, `origin=ai-generated` + embedded provenance.

**Why this stack:** SDXL/Qwen on OpenVINO sidesteps the fp16 XPU bug (#5220);
face-swap reuses code we already ship; commercial licensing stays clean with
SDXL/Qwen/FLUX-schnell (avoid FLUX-dev for commercial).

**Surface:** `POST /api/sets/{id}/generate-image` (kind = faceswap | scene | edit),
async job (calls the generation service), result → `generated/`, `origin=ai`; a
"Generate image" control in `set_detail.html`; a `media set generate-image` CLI.

**Effort:** Med (stand up + call the OpenVINO/ComfyUI service); LoRA phase High.

**Risks:** #5220 (use OpenVINO); training-free identity is "good not perfect"
(LoRA fixes it, at training cost); licensing per base model; ethics (see below).

---

## Plan B — Video (morph: extrapolate frames BETWEEN set items)

**Goal:** build a video by **synthesizing the in-between frames between the set's
items** — a morph that flows photo→photo→photo as motion. NOT a slideshow (static
stills + crossfades). The set's images are the keyframes; the engine invents/derives
the frames between each consecutive pair, then all frames are assembled to mp4
(hardware-encoded on the B70).

**Engine — Wan 2.2 FLF2V (generative), on the B70 via ComfyUI:**
- **Per-pair generation.** Order the set members, then for each consecutive pair
  (A→B, B→C, …) send the two images to **Wan 2.2 FLF2V** (first-frame/last-frame)
  and let it *generate* the in-between frames — real synthesized motion, not a
  pixel warp. Runs in the Intel `llm-scaler` ComfyUI-XPU container.
- **Transport.** The app calls ComfyUI over HTTP (`gen_service.py`): upload the
  two frames, submit the FLF2V workflow (a template with the two image inputs +
  params substituted), poll to completion, fetch the output clip/frames.
- **Assembly.** Concatenate the per-pair clips (or their frames via
  `set_render.frames_to_video`) into the final morph, ffmpeg-encoded with **VAAPI
  hardware HEVC on the B70** (software fallback). Output → `generated/…`,
  `origin=ai` + provenance (`model=wan2.2-flf2v`).

**Why:** FLF2V is the "best results" morph (true generated motion). It's a
different stack from the image path on purpose — a ComfyUI service on the B70,
not the OpenVINO diffusers path.

**Surface:** `POST /api/sets/{id}/generate-video` (async job w/ progress polling,
reusing the pattern-index/train job pattern), result → `generated/`; a "Generate
video" control in `set_detail.html`; a `media set generate-video` CLI.

**Effort:** High — the app-side integration (service client + morph pipeline) is
buildable/testable now; the B70 side (container + Wan models + workflow) is a
documented deploy step, and generation is minutes/clip.

**Risks:** FLF2V on Battlemage is genuinely experimental (fp64 nodes, kernel
compiles) — budget failures; identity can drift across a long morph chain.

---

## Shared architecture

- **Generation service, not in-process.** Heavy models run in a **separate local
  service** (an OpenVINO GenAI process and/or a ComfyUI-XPU instance) that the
  FastAPI app calls over HTTP as an **async job** — the same shape as the media
  worker (`worker_client`/`worker_server`) we already run. Load models once
  (JIT-compile cost is real). The web request returns a job id; the UI polls.
- **Deterministic engines run in-process** (Pillow) or as a short ffmpeg
  subprocess — no service needed, so Phase 1 has zero new infrastructure.
- **Outputs** are written to `generated/`, recorded in the DB with `origin` +
  source set + technique + params, excluded from normal scans, and shown with the
  appropriate label/badge.

## Licensing & ethics

- **Licensing:** prefer Apache/OpenRAIL bases (SDXL, FLUX-schnell, FLUX.2-Klein,
  **Qwen-Image-Edit**). Non-commercial (self-host personal use only): FLUX.1-dev /
  Kontext-dev, `inswapper`, LivePortrait. Audit every model + restorer asset.
- **Ethics (person generation is deepfake-capable):** gate generation to the
  account owner's own sets; keep any bundled NSFW filter **on**; embed the
  AI-generated provenance mark (above) and consider a visible watermark;
  audit-log who generated what. Decide policy before enabling Phase 2 on
  person-sets.

## Decided: video engine = Wan 2.2 FLF2V (2026-09-17)

The morph is built with **generative** motion, not classical interpolation: for
each consecutive pair of set members (A→B, B→C, …), **Wan 2.2 FLF2V** is given the
two images as first/last frame and generates the in-between frames; the per-pair
clips are concatenated into the final morph. Runs via a **ComfyUI (Intel
`llm-scaler`) service on the B70**; the app talks to it over HTTP (see
`gen_service.py`). Highest realism, accepted trade-offs: minutes/clip and an
experimental Arc stack. RIFE was declined.

**B70 deploy prerequisites** (documented, not automatable from the app): run the
llm-scaler ComfyUI-XPU container and download the Wan 2.2 FLF2V models, then in
ComfyUI build a Wan FLF2V graph (two `LoadImage` nodes = first/last frame, a
positive prompt, a `SaveImage` output), **Save (API Format)**, and drop that JSON
at `<library>/.media/flf2v.api.json`. **No JSON editing:** the app auto-detects
the two `LoadImage` nodes (lowest id = first frame) and the positive
`CLIPTextEncode`, and fills them per pair — you keep whatever steps/frames/seed
you baked into the graph. (Advanced: if you'd rather wire the inputs by hand, put
`__FIRST_IMAGE__ / __LAST_IMAGE__ / __PROMPT__ / __FRAMES__ / __SEED__` tokens in
the JSON and those are substituted instead.) No env vars needed co-located —
`MEDIA_GEN_SERVICE_URL` defaults to `http://127.0.0.1:8188` and the workflow to
that `.media/` path; set them only to override.

## Status

- [x] Support layer: `generated/` excluded from scans (fast_scan/scanner);
      `generated_artifacts` registry (manual.db) with `origin` (composite|ai);
      provenance-mark helpers (EXIF/PNG-text + ffmpeg `-metadata`) in `set_render.py`
- [x] Video morph (Wan FLF2V) app-side integration — `gen_service.py` (ComfyUI
      client), `set_video.morph_from_set` (per-pair FLF2V → assemble), `media set
      generate-video` CLI, and the **set view**: 🎬 button in `set_detail.html`,
      `POST /api/sets/{id}/generate-video` (async, `_spawn_job`) + `/status` +
      `/api/sets/{id}/generated` + `GET /generated/{artifact_id}`. Verified against
      a mock ComfyUI; the real Wan model runs on the **B70 box**.
- [ ] Images — face-swap (`inswapper`, reuse InsightFace) then InstantID/Qwen via
      OpenVINO/ComfyUI (**B70 box**)
- [ ] `/generated` gallery page + "AI" badge on cards (copy the trash-badge pattern)
- [ ] Per-set LoRA training (**B70 box**)

_Dropped: the slideshow / Ken Burns / collage / contact-sheet composites — not the
feature (that's static stills, not generated/morphed media)._
