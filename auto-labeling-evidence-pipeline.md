---
title: Auto-labeling & the Evidence Pipeline
---

# Auto-labeling & the Evidence Pipeline

How foundation models turn raw video into structured, searchable, reviewable annotations, and
why the mature systems treat model output as **evidence for a human**, not automatic ground truth.

---

## 1. The problem: labeling video does not scale by hand

A minute of manipulation video at 30 fps is 1,800 frames. Hand-labeling masks, boxes, and tracks
for hands, tools, objects, and contacts across thousands of episodes is impossible manually. So
the modern pipeline flips it: **let foundation models draft the annotations, and put humans on
review instead of on labeling from scratch.** The human's time moves from "draw every mask" to
"accept, correct, or reject the model's drafts, and only for the hard cases."

That single move (model drafts, human reviews) is the backbone of every model-assisted annotation
system.

## 2. Evidence, not ground truth (the design philosophy)

The subtle and important part: a mature pipeline does **not** let the model's output become the
label automatically. Model outputs are treated as **evidence and routing signals**:

- A mask is a *proposal* with a confidence, not a fact.
- An embedding similarity is a *hint* that two clips match, not a decision.
- An uncertainty flag *routes* a hard case to a human, it does not resolve it.

Why this matters: physical-AI training data feeds safety-relevant policies, so silent label
errors are expensive. Keeping model outputs as inspectable evidence (with provenance) means a
human can always audit *why* a label exists, and high-risk cases never bypass review. The system's
job is to make the evidence **reliable, searchable, and reviewable**, not to replace the human.

## 3. The model stack (the vocabulary)

The pieces that draft the evidence, grouped by what they produce:

**Promptable segmentation (masks).**
- **SAM / SAM 2 / SAM 3** (Segment Anything) — prompt with a point, box, or text and get a mask;
  SAM 2 adds video (masklets that persist across frames). This is the workhorse for masks and
  tracks of hands, tools, and objects.
- **Grounding DINO** — open-vocabulary detection: give a text phrase ("the blue connector"), get
  boxes. Often chained into SAM to turn a phrase into a mask ("grounded segmentation").

**Dense visual features (similarity, quality, OOD).**
- **DINOv2 / DINOv3** — self-supervised features that are excellent for similarity, mask-quality
  scoring, scene features, and out-of-distribution checks, without task labels.

**Tracking (identity over time).**
- **ByteTrack** (detection-based multi-object tracking), **CoTracker / TAPIR** (dense point
  tracking), **XMem** (long-video mask propagation). These maintain object IDs and track
  continuity, and their failures (drift, ID switches) become uncertainty flags.

**Video-text embeddings (search and dedup).**
- **CLIP-style** and video-native embeddings (e.g. Cosmos-Embed) map clips into a vector space
  for clip search, inverse video search, semantic deduplication, scenario mining, and
  hard-negative mining.

## 4. From model outputs to durable artifacts

The engineering value is not running the model once; it is turning each output into a **durable,
versioned artifact** that downstream systems and human reviewers can trust:

- masks, boxes, tracks, object IDs, confidence scores,
- dense features, clip embeddings, similarity links, dedup flags,
- evidence spans (which frames support which claim), uncertainty and failure cues.

And every one is **versioned** with its checkpoint, config, preprocessing, prompt/query, frame
range, artifact hash, and storage reference. That provenance is what makes the evidence auditable
and the results reproducible. It is the difference between "the model said so" and "here is
exactly which model, prompt, and frames produced this, and here is the human who confirmed it."

## 5. Retrieval is the quiet backbone

Almost every useful operation on a large video corpus is a **similarity search** in embedding
space, which is why vector databases (FAISS, Milvus, Qdrant, Weaviate) sit at the center:

```python
# Sketch: index clip embeddings, then search for similar / duplicate clips.
import numpy as np, faiss

emb = np.stack(clip_embeddings).astype("float32")   # (N, D), L2-normalized
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])             # inner product == cosine on unit vectors
index.add(emb)

# find near-duplicates of clip q (semantic dedup / scenario mining)
D, I = index.search(emb[q:q+1], k=10)               # top-10 most similar clips
duplicates = [i for i, d in zip(I[0], D[0]) if d > 0.95 and i != q]
```

The same primitive powers **dedup** (too-similar clips), **scenario mining** (find all clips like
this rare event), **hard-negative mining** (near-misses for contrastive training), and
**gold-sample selection** (representative clips for a human to label as the reference). This is
the exact dot-product similarity from [attention](transformers.html) and
[InfoNCE](activations-losses-optimizers.html), applied to whole video clips.

## 6. Uncertainty and routing (where humans come in)

The system earns its keep by knowing when *not* to trust itself. Failure and uncertainty flags
are raised for: occlusion, hidden hands, weak track continuity, fast motion, camera shake, object
confusion, poor visibility, track drift, and ambiguous contact. Each flag **routes** the case to a
human queue rather than silently committing a label. Good routing is what lets a small review team
cover a large corpus: humans spend their attention only where the model is unsure or the stakes
are high.

## 7. Inspection tooling

Reviewers need to *see* the evidence over time, so the pipeline exposes visualization and review
hooks in **Rerun** and **Foxglove**, over multimodal logs commonly stored as **MCAP** (a container
for synchronized, timestamped robotics streams). The reviewer scrubs the video with masks, tracks,
and flags overlaid, and accepts / corrects / rejects efficiently. Making evidence *inspectable* is
as much of the job as generating it.

## 8. The shape of the whole pipeline

```
raw egocentric / robot video
        │
        ▼
  foundation models draft evidence
  (SAM/DINO masks+tracks, DINO features, CLIP/Cosmos embeddings)
        │
        ▼
  durable versioned artifacts  ──►  vector DB (search / dedup / mining)
  (masks, tracks, embeddings, confidences, provenance)
        │
        ▼
  uncertainty + failure flags  ──►  route hard/high-risk cases to humans
        │
        ▼
  human review in Rerun / Foxglove  (accept / correct / reject)
        │
        ▼
  trusted, policy-fit training + evaluation data
```

---

*The one-line summary: foundation models (SAM, DINO, video-text embeddings) draft masks, tracks,
and embeddings; the system stores them as versioned, searchable evidence with provenance, flags
its own uncertainty to route hard cases to humans, and never lets model output silently become
ground truth. The quiet backbone is embedding-space similarity search.*
