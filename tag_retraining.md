# Teaching CLIP new concepts — options

## The core problem

The `/tags` CLIP "training" is a **linear probe**, not fine-tuning. For each tag it
fits a single `Linear(512 → 1)` logistic regression over the tag's confirmed/rejected
photos' **frozen** CLIP embeddings (`clip_tag_classifier.fit_from_examples`), producing
a 512-dim weight vector. Inference is `sigmoid(embedding · w + b)` — a dot product.
CLIP itself never moves.

This works great **when CLIP's embedding already encodes the distinguishing feature**.
It fundamentally **cannot** work when the embedding doesn't — no classifier on top of a
frozen embedding can recover information the embedding never captured. Adding more
examples does nothing in that case.

**Likely root cause for us:** OpenAI's CLIP (`ViT-B-32/openai`) was trained on a
*heavily filtered* web dataset (NSFW / aesthetic / "undesirable" content removed). So
niche / adult / private concepts weren't "missed" — they were **omitted from training**.
That's why more examples don't help: the embedding never learned to represent them.

## Why per-tag (not "one model over all tags")

For a **linear probe**, a joint `Linear(512 → N)` head is mathematically identical to N
independent `Linear(512 → 1)` heads — the output rows share no parameters. "Training all
tags together" buys **nothing** unless you add real weight-sharing (a hidden layer) or
fine-tune the backbone. Meanwhile per-tag wins on:
- **Incremental retraining** — retrain only the tag you changed, in seconds.
- **Sparse labels** — we only have per-tag confirm/reject; most photos are unlabeled for
  most tags. A joint multi-label model wants a dense label matrix we don't have.

So per-tag is correct **for the current design**. The real question is the *features*,
not how the head is trained.

## The options ladder (cheapest → heaviest)

### 1. Swap the backbone to a LAION-trained OpenCLIP  ← recommended first move
Not "training," but the highest-leverage action. LAION was **not** NSFW-filtered the way
OpenAI's set was, so LAION-trained OpenCLIP models (`ViT-L-14` / `ViT-H-14` on `laion2b`,
or SigLIP) **already learned the concepts we're missing** — and being bigger/newer, they
see more of everything.
- **Cost:** changes embedding dim/space → **re-embed the whole library once** (bigboy has
  the headroom) + bump the embeddings schema. One-time migration that upgrades *every*
  feature (similarity, region/tile search, set/category centroids, tags) — not just tags.
- For most "CLIP never saw it" concepts, **this alone fixes it.**

### 2. Add a complementary encoder for tags only (concatenated)
Keep the current CLIP as the shared substrate (nothing else breaks), but feed the tag
classifier `[CLIP ⊕ DINOv2]` or `[CLIP ⊕ LAION-CLIP]`. DINOv2 (self-supervised, unfiltered)
captures fine-grained visual detail CLIP collapses. Linear probe stays per-tag/incremental;
it just gets features that actually encode the concept. Lower blast radius than swapping
the shared backbone.

### 3. LoRA / adapter fine-tune the vision encoder  ← the real "training CLIP"
For concepts so bespoke that even a stronger off-the-shelf backbone misses them. Add small
trainable LoRA adapters to a frozen OpenCLIP vision transformer, train on our tag
confirm/reject examples (supervised — pull same-concept images together). **LoRA, not full
fine-tune**, because we have tens of examples, not millions (full fine-tune overfits).
- Store as a **separate tag encoder** so it doesn't disturb the shared embedding.
- **Offload training to bigboy** — slots into the existing per-tag training job/poll/
  checkpoint machinery we built for the YOLO fine-tune.

### Already covered: localizable objects
For concepts that are **objects with a bounding box**, the per-tag **YOLO-World fine-tune**
already teaches a new class from our boxes (and is offloaded to bigboy). The gap this doc
addresses is **whole-image / abstract** concepts.

## Recommended path

1. **Backbone swap first** (option 1). Highest value, no training, upgrades the whole app.
2. **LoRA fine-tune** (option 3) only for the long tail that survives a stronger backbone.

Don't do option 3 on the current `ViT-B-32` — that's teaching a weak, small model things a
better model already knows.

## Diagnostic gate (run before committing to a re-embed)

Take 2–3 of the worst tags, embed their example images with both the current model and a
LAION `ViT-L-14`, and measure whether LAION **separates** them (linear-probe CV accuracy /
cluster tightness) where the current model can't.
- LAION separates them → option 1 is enough; do the backbone swap + re-embed.
- Even LAION can't → you need option 3 (LoRA fine-tune).

Runs on bigboy, touches nothing in the app.
