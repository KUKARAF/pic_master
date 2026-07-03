# Features TODO

## Phase 1: File Inventory (done)
- [x] `media init` — create `.media/` repo
- [x] `media add` — fast file discovery via GNU find, store metadata in SQLite
- [x] `media commit` — xxhash all untracked files
- [x] `media status` — show unhashed files + moved file candidates
- [x] `media mv` — record a file move without re-hashing
- [x] `media ls / count / hashes` — query the DB
- [x] `media find_broken` — detect corrupted images/videos
- [x] `.mediaignore` support (git-ignore syntax)

## Phase 2: CLIP Image Search (in progress)
- [x] `embeddings` table in SQLite (file_id, embedding BLOB, model, indexed_at)
- [x] `media index [path]` — run CLIP (ViT-B-32), store float32 embeddings
- [x] `media search <query>` — encode text with CLIP, cosine similarity over all embeddings
- [ ] Install and test end-to-end (`pip install -e .` with open-clip-torch)
- [ ] Progress bar for `media index` (reuse HashEstimator pattern)
- [ ] `media index --reindex` flag to force re-embedding already indexed files
- [ ] `--model` flag wired through CLI (allow swapping ViT-B-32 for larger models)

## Phase 3: Auto-Captioning
- [ ] Pick a vision LLM backend (LLaVA local, or Gemini/GPT-4V API)
- [ ] `captions` table — (file_id, caption TEXT, model, created_at)
- [ ] `media caption [path]` — generate and store captions
- [ ] Use captions as training data for CLIP finetuning (Phase 5)
- [ ] Expose captions in `media search` results

## Phase 4: Face & Similarity Search
- [ ] Decide on face detection backend (InsightFace, DeepFace, or YOLO + face model)
- [ ] `faces` table — (id, file_id, bbox, embedding BLOB, identity TEXT)
- [ ] `media faces [path]` — detect and embed all faces
- [ ] `media search --faces <image>` — find photos containing the same face
- [ ] Clustering faces by identity (unsupervised, e.g. DBSCAN on face embeddings)
- [ ] `media who <image>` — show which identity cluster(s) appear in a photo

## Phase 5: CLIP Finetuning
- [ ] Generate (image, caption) pairs from Phase 3 captions
- [ ] Implement Tier 3: sentence-transformers contrastive finetuning
  - Use same-event photos as positives, different-event as negatives
  - Triplets derivable from folder structure / timestamps in existing DB
- [ ] `media train [--epochs N] [--output MODEL_PATH]` — finetune on local data
- [ ] `media index --model <finetuned_model>` — re-index with custom model
- [ ] Evaluate: compare search quality before/after finetuning
- [ ] Optional Tier 2: LoRA adapters for deeper finetuning (needs 8GB+ VRAM)

## Phase 6: Web UI (in progress)
- [x] `media web [--host] [--port]` — FastAPI server on localhost:8000
- [x] Gallery grid with thumbnails, tag chips, ⭐ "i am feeling lucky" similar-image button
- [x] Photo detail page — full image, metadata, live tag add/remove
- [x] CLIP text search (`/search?q=`)
- [x] Tag search (`/search?tag=`)
- [x] Similar images page (CLIP cosine similarity, top 20)
- [x] Tags API (add, remove, list, search by tag)
- [x] Thumbnail generation + cache in `.media/thumbs/`
- [x] Faces stub (placeholder UI until Phase 4 is done)
- [ ] Face search UI (`/search?face_id=`) once Phase 4 is implemented
- [ ] Pagination on gallery (currently limited to 200 files)
- [ ] Upload / drag-and-drop to add files via UI
- [ ] Tag autocomplete in add-tag input

## Infrastructure / Nice-to-haves
- [ ] `media stats` — summary of DB (total files, indexed, broken, faces, etc.)
- [ ] `media export` — dump hash manifest to text (for external verification)
- [ ] Batch size / worker count auto-tuned based on available CPU/GPU
- [ ] `--json` output flag on search/ls for scripting
