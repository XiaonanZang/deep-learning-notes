---
title: Vision Transformers (ViT)
---

# Vision Transformers (ViT)

How the transformer encoder — built for text in the
[Transformer Architecture](transformers.html) note — is applied to images with almost
no changes.

---

## 1. Core idea

ViT asks: what if an image were a sequence of patches? Feed those patch embeddings into a
standard transformer encoder, just as you would feed word tokens into a language model.
No convolutions needed.

---

## 2. The classical example: image classification on ImageNet

**Input:** a 224×224 RGB image.

**Step 1 — patch tokenization.** Divide the image into non-overlapping 16×16 patches.

```
Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches
Each patch:        16 × 16 × 3 = 768 raw pixel values
```

**Step 2 — linear projection.** Each patch is flattened and projected to the model
dimension D (typically 768 for ViT-Base):

```
Patch tokens:  (B, 196, 768)
```

**Step 3 — [CLS] token.** A learnable classification token is prepended to the sequence.
Its final hidden state is used for classification:

```
Sequence:  (B, 197, 768)   [196 patches + 1 CLS token]
```

**Step 4 — positional embeddings.** Learned 1D positional embeddings are added (one per
position) so the model knows patch spatial order:

```
X = patch_embeddings + pos_embeddings   →   (B, 197, 768)
```

**Step 5 — transformer encoder.** Standard stack of L encoder layers (ViT-Base: L=12,
h=12 heads, D=768). Each patch attends to every other patch globally, with no inductive
bias about spatial locality.

**Step 6 — classification head.** Take the CLS token output at position 0, pass through a
linear head:

```
CLS output:  (B, 768)   →   Linear(768, num_classes)   →   (B, 1000)
```

---

## 3. Shape flow end to end

```
Image:          (B, 3, 224, 224)
  → patches:    (B, 196, 768)      flatten 16×16×3 per patch
  → + CLS:      (B, 197, 768)      prepend learnable token
  → + pos emb:  (B, 197, 768)      add positional embeddings
  → encoder ×L: (B, 197, 768)      L transformer layers
  → CLS out:    (B, 768)           index position 0
  → head:       (B, 1000)          linear classifier
```

---

## 4. The takeaway: attention is attention

ViT proves that the same self-attention block that resolves "The animal... it" in text
also reads patch-to-patch relationships in images. Other domains reuse the identical
primitive over different token types — a time-series model attends over timesteps, a graph
transformer attends over nodes.

**One sentence to remember:** *patches, timesteps, and nodes are all just tokens fed into
the same attention mechanism.*

---

[← Transformer Architecture](transformers.html) · [Back to contents →](index.html)
