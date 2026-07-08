---
title: Vision Transformers (ViT)
---

# Vision Transformers (ViT)

How the transformer encoder — built for text in the
[Transformer Architecture](transformers.html) note — is applied to images with almost
no changes. The plain-language tour comes first, then a from-scratch, fully commented
PyTorch implementation that runs end to end, from a raw image tensor to class logits.
Only the ViT-specific machinery (patchify, CLS token, positional embeddings) is built
here; the attention internals live in the transformer note, so this note uses a standard
encoder for that stack and stands on its own as the source of truth for ViT.

---

## 1. Core idea

ViT asks: what if an image were a sequence of patches? Cut the image into a grid of
small patches, turn each patch into a vector, and feed that sequence into a standard
transformer encoder — exactly as you would feed word tokens into a language model. No
convolutions needed. The encoder does not know or care that the tokens came from pixels;
a token is a token.

Everything that follows is just the plumbing that turns pixels into tokens, plus one
extra token (the CLS token) that carries the image's summary.

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

**Step 3 — [CLS] token.** A single learnable classification token is prepended to the
sequence. Its final hidden state is used for classification:

```
Sequence:  (B, 197, 768)   [196 patches + 1 CLS token]
```

*Why a CLS token instead of just pooling the patches?* Both work — a common alternative
is to mean-pool all 196 patch outputs and classify that. The CLS token is a *learnable
aggregator*: it starts as one trainable vector, and through every attention layer it
attends to all patches and decides, via learned weights, how to summarize them. Mean
pooling forces a fixed, uniform "average everything" summary; the CLS token can learn a
task-specific, weighted summary (attend more to the foreground object, less to
background). It also keeps the classification pathway cleanly separated from the patch
representations. The cost is one extra token of compute and a slight asymmetry (position
0 is special). In practice CLS and mean-pooling perform similarly; ViT chose CLS to mirror
BERT's `[CLS]` design.

**Step 4 — positional embeddings.** Learned 1D positional embeddings are added (one per
position) so the model knows patch spatial order:

```
X = patch_embeddings + pos_embeddings   →   (B, 197, 768)
```

*Why learned, not sinusoidal?* The transformer note used fixed **sinusoidal** encodings
(a closed-form function of position, zero parameters). ViT instead **learns** one vector
per position — a `(197, 768)` table trained by gradient descent. Both inject order and
both work; learned embeddings give the model freedom to discover whatever positional
structure helps, at the cost of extra parameters and a table tied to a *fixed* sequence
length. That fixed length is the catch: sinusoidal encodings extend to any length for
free, but a learned table has exactly 197 rows. **What breaks at an unseen resolution:**
feed a 384×384 image (576 patches, not 196) and the 197-row table no longer has enough
positions. The standard fix is to **interpolate** the learned position table to the new
grid size (2D bilinear interpolation over the 14×14 layout) — a routine but necessary step
whenever you fine-tune ViT at a higher resolution than it was pretrained on.

**Step 5 — transformer encoder.** Standard stack of L encoder layers (ViT-Base: L=12,
h=12 heads, D=768). Each patch attends to every other patch globally, with no inductive
bias about spatial locality.

*Why "no locality bias" is the defining tradeoff.* A CNN is hard-wired to look at local
neighborhoods first (a 3×3 kernel only sees adjacent pixels) and to treat every location
the same way (weight sharing). Those are strong, correct priors about images baked into
the architecture — so a CNN learns efficiently from modest data. A ViT has **none** of
that: at layer 1, patch (0,0) can attend to patch (13,13) just as easily as to its
neighbor, and it must *learn* from data that nearby patches usually matter more. That
freedom is why **ViT is data-hungry** — it underperforms CNNs on ImageNet-scale data alone
and only pulls ahead when pretrained on very large datasets (JFT-300M and beyond), where it
can learn the priors a CNN was given for free. This is the same tradeoff you will meet
again in graphs: a **GCN** with a fixed neighborhood (aggregate only from graph neighbors)
is the CNN analog, while a **graph transformer** (every node attends to every node) is the
ViT analog — more flexible, more data-hungry. *Attention buys flexibility by giving up
built-in priors; you pay for it in data.*

**Step 6 — classification head.** Take the CLS token output at position 0, pass through a
linear head:

```
CLS output:  (B, 768)   →   Linear(768, num_classes)   →   (B, 1000)
```

---

## 3. Shape flow end to end

```
Image:          (B, 3, 224, 224)
  → patches:    (B, 196, 768)      flatten 16×16×3 per patch, project to D
  → + CLS:      (B, 197, 768)      prepend learnable token
  → + pos emb:  (B, 197, 768)      add learned positional embeddings
  → encoder ×L: (B, 197, 768)      L transformer layers (same block as the text note)
  → CLS out:    (B, 768)           index position 0
  → head:       (B, 1000)          linear classifier
```

---

## 4. PyTorch implementation — step by step

The only ViT-specific code is the front end that turns an image into a `(B, 197, 768)`
token sequence, plus the classification head. The encoder in between is a standard
transformer encoder — the same architecture built from scratch in the
[Transformer Architecture](transformers.html) note. This note stays focused on its own
topic (how images become tokens) and uses PyTorch's built-in encoder for the attention
stack, which keeps the whole thing runnable end to end on its own.

```python
import torch
import torch.nn as nn
```

---

### Step 1: image → patch tokens

We cut the image into non-overlapping patches and turn each patch into one D-dimensional
vector. The clean trick is a single `Conv2d` whose kernel size **and** stride both equal
the patch size: a stride-P, size-P convolution touches each patch exactly once with no
overlap, and its `out_channels` do the linear projection to D in the same op. This fuses
Step 1 (patchify) and Step 2 (project) of the conceptual tour into one layer.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_chans=3, D=768):
        super().__init__()
        self.patch = patch
        # num_patches along one side, then squared for the full grid.
        # 224 / 16 = 14 per side → 14 * 14 = 196 patches total.
        self.grid  = img_size // patch                 # 14
        self.n_patches = self.grid * self.grid         # 196

        # kernel_size = stride = patch is the key line. A normal conv slides with
        # stride 1 and overlaps; setting stride = kernel makes each placement land on a
        # fresh, non-overlapping patch. out_channels = D means each patch is projected
        # to a D-vector — the linear projection of Step 2, done by the conv weights.
        self.proj = nn.Conv2d(in_chans, D, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        # Conv with stride=patch produces one D-vector per patch, laid out on a 2D grid:
        #   (B, 3, 224, 224) → (B, D, 14, 14)
        # The 14x14 spatial map IS the patch grid; channel axis D is the patch embedding.
        x = self.proj(x)                    # (B, D, 14, 14)

        # Flatten the 2D grid into a 1D sequence of tokens.
        # flatten(2) collapses the last two axes (14, 14) → 196, giving (B, D, 196).
        # transpose(1, 2) then swaps to (B, 196, D) so the token axis is in the middle,
        # matching the (B, T, D) convention the encoder expects.
        x = x.flatten(2).transpose(1, 2)    # (B, 196, D)
        return x
```

---

### Step 2: prepend the [CLS] token

The CLS token is one learnable D-vector, shared across all images, copied to every item
in the batch and placed at position 0. After the encoder runs, position 0 is the image
summary we classify.

```python
class AddCLS(nn.Module):
    def __init__(self, D=768):
        super().__init__()
        # nn.Parameter registers this as a trainable weight. Shape (1, 1, D):
        # the two leading 1s are placeholders for (batch, sequence) so it broadcasts /
        # expands cleanly to any batch size. Only one CLS vector is ever learned.
        self.cls = nn.Parameter(torch.zeros(1, 1, D))

    def forward(self, x):
        # x: (B, 196, D)
        B = x.shape[0]

        # expand (not repeat) copies the single CLS vector to every batch item WITHOUT
        # allocating new memory — it just sets stride 0 on the batch axis. Result is a
        # (B, 1, D) view of the same underlying parameter.
        cls = self.cls.expand(B, -1, -1)    # (B, 1, D)

        # Concatenate along the token axis (dim=1): CLS goes first, then the 196 patches.
        # 196 + 1 = 197 tokens. CLS is now position 0.
        x = torch.cat([cls, x], dim=1)      # (B, 197, D)
        return x
```

---

### Step 3: add learned positional embeddings

Attention is permutation-invariant, so without a position signal the patch grid would be
a bag of patches. ViT adds a *learned* table with one row per position (including the CLS
slot), unlike the fixed sinusoids of the text note.

```python
class AddPosEmbed(nn.Module):
    def __init__(self, n_tokens=197, D=768):
        super().__init__()
        # One trainable vector PER position. Shape (1, 197, D): leading 1 broadcasts over
        # the batch. Contrast with sinusoidal_encoding() in the transformer note, which is
        # a fixed function of position with zero parameters. Here the 197 rows are learned
        # — which is why this table is tied to exactly 197 positions (see the resolution
        # note in §2: a different image size needs this table interpolated to a new grid).
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, D))

    def forward(self, x):
        # x: (B, 197, D). Broadcasting adds the (1, 197, D) table to every batch item.
        return x + self.pos                 # (B, 197, D)
```

---

### Step 3b: interpolating the position table for a new resolution

The learned table has **exactly** as many rows as the pretraining grid (197 = 14×14 patches +
CLS). Fine-tune at a higher resolution and the patch-embed still works — it is applied per patch —
but you now have *more* patches than rows in the table. The fix interpolates the **position
embeddings**, not the image: view the 196 patch vectors as a 14×14 grid of D-dim vectors and
2D-resize that grid to the new patch layout. The CLS row has no spatial location, so it is kept
aside and re-attached unchanged.

```python
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, old_hw, new_hw):
    # pos_embed: (1, 1 + old_h*old_w, D) — the LEARNED table (CLS row first).
    # old_hw / new_hw: (h, w) patch-grid sizes, e.g. (14,14) -> (28,28).
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    D = pos_embed.shape[-1]

    cls_pe  = pos_embed[:, :1]                  # (1, 1, D)  CLS — no spatial location, keep as-is
    grid_pe = pos_embed[:, 1:]                  # (1, old_h*old_w, D)  the patch positions

    # Recover the spatial grid, then put D on the CHANNEL axis. F.interpolate expects a fixed
    # layout (N, C, H, W): it leaves N and C alone, resizes only the LAST TWO axes, and resizes
    # each channel INDEPENDENTLY (it blends across space, never across channels).
    # Mental model: we are resizing a 14x14 "image" whose per-pixel value is a D-dim vector —
    # so D plays the role of channels. Hence D must move into dim 1, and the (old_h, old_w) grid
    # must be the trailing spatial dims. If D were left last, interpolate would resize (old_w, D)
    # instead — crushing the 768 feature dims to the new size and mixing unrelated embedding
    # components. The permute does no math; it only satisfies interpolate's channels-first contract.
    grid_pe = grid_pe.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)   # (1, D, old_h, old_w)

    # The actual resize: each of the D feature-planes is upsampled over the patch grid, and each
    # new location's PE is a bilinear blend of its 4 nearest old PEs.
    grid_pe = F.interpolate(grid_pe, size=(new_h, new_w),
                            mode='bilinear', align_corners=False)        # (1, D, new_h, new_w)

    # Flatten the new grid back to a token sequence and re-attach CLS.
    grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)   # (1, new_h*new_w, D)
    return torch.cat([cls_pe, grid_pe], dim=1)                           # (1, 1 + new_h*new_w, D)

# 224px (14x14) pretrain -> 448px (28x28) fine-tune
pos_embed = torch.randn(1, 197, 768)
new_pe = interpolate_pos_embed(pos_embed, (14, 14), (28, 28))
print(pos_embed.shape, "->", new_pe.shape)   # torch.Size([1, 197, 768]) -> torch.Size([1, 785, 768])
```

Why *bilinear* works at all: learned position embeddings come out **spatially smooth** (adjacent
patches learn similar vectors), so blending between them yields sensible embeddings for the new
in-between locations. It is a warm start, not an exact answer — fine-tuning then adapts them.
(The original ViT/DeiT papers use **bicubic**, a smoother 16-neighbour kernel; bilinear is the
same idea with 4 neighbours.)

**Could this use `transpose` + `view` instead of `permute` + `reshape`?** Yes, with two caveats —
and both are the exact rules from the encoder note's Step 5:

- **`permute` vs `transpose`:** `transpose` swaps *exactly two* axes; `permute` reorders *all* at
  once. The `(B,H,W,D) → (B,D,H,W)` move is a 4-axis reorder, so it takes **two** transposes to
  equal one permute: `.transpose(1, 3).transpose(2, 3)`. Same result (verified identical), but
  `permute(0, 3, 1, 2)` says it in one line.
- **`reshape` vs `view`:** `view` needs a **contiguous** (compatible-stride) tensor and errors
  otherwise; `reshape` silently falls back to a copy when needed. `permute`/`transpose` return a
  **non-contiguous** view, so the flatten-back step with `view` requires an explicit
  `.contiguous()` first — `.permute(0, 2, 3, 1).contiguous().view(1, new_h*new_w, D)`. `reshape`
  does that for you, which is why it is the safe default here.

---

### Step 4: full ViT (front end + encoder + head)

Now assemble the pieces: patch embed → CLS → pos embed → a standard transformer encoder →
take position 0 → linear head. The encoder is the same architecture as the text note; we
use PyTorch's built-in version here so this note runs end to end on its own.

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch=16, in_chans=3,
                 D=768, h=12, num_layers=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, D)
        n_tokens = self.patch_embed.n_patches + 1        # +1 for the CLS token → 197
        self.add_cls = AddCLS(D)
        self.add_pos = AddPosEmbed(n_tokens, D)

        # A standard transformer encoder — the SAME block derived from scratch in the
        # transformer note (multi-head attention + FFN + residual + LayerNorm). Here we use
        # PyTorch's built-in so this note stays focused on the ViT-specific parts and still
        # runs. Flags chosen to match the from-scratch note exactly:
        #   batch_first=True   → expects (B, T, D), the convention we built above
        #   norm_first=True    → pre-norm order (LayerNorm before each sub-layer)
        #   dim_feedforward=4D → the 4x FFN expansion
        #   activation="gelu"  → GELU, as in the note
        # It takes (B, 197, D) and returns (B, 197, D); it has no idea T came from patches.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=h, dim_feedforward=4 * D,
            activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor=False just silences a benign perf warning that PyTorch
        # prints when norm_first=True; it does not change the math.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.norm = nn.LayerNorm(D)                       # final norm before the head
        self.head = nn.Linear(D, num_classes)             # CLS vector → class logits

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.patch_embed(x)             # (B, 196, D)
        x = self.add_cls(x)                 # (B, 197, D)
        x = self.add_pos(x)                 # (B, 197, D)

        x = self.encoder(x)                 # (B, 197, D) — global attention over patches

        # Take ONLY the CLS token (position 0) as the image summary, then classify.
        # x[:, 0] indexes the token axis at 0 → (B, D). This is where the "aggregator"
        # role of CLS pays off: one vector now stands in for the whole image.
        cls_out = self.norm(x[:, 0])        # (B, D)
        logits  = self.head(cls_out)        # (B, num_classes)
        return logits
```

---

### Step 5: end-to-end smoke test

```python
# --- Smoke test ---
B           = 2                 # batch of 2 images
img_size    = 224
num_classes = 1000

# Random image batch standing in for real pixels.
images = torch.randn(B, 3, img_size, img_size)          # (B, 3, 224, 224)

model  = ViT(img_size=img_size, patch=16, D=768, h=12,
             num_layers=12, num_classes=num_classes)
logits = model(images)

print(logits.shape)             # expect: torch.Size([2, 1000])
assert logits.shape == (B, num_classes)
```

---

## 5. Two things called "embedding": input projection vs output latent

"Embedding" is overloaded. In a ViT it names two different tensors at opposite ends of the
network, and it is worth pinning down which is which — because when people say a "CLIP
embedding" or "DINO features," they mean the *second* one, not the first.

**(a) Input embedding — the front-end projection.** The `PatchEmbed` + CLS + pos stack that
turns pixels into the `(B, 197, D)` token sequence. In a *language* model this is the
`nn.Embedding` lookup table (token ID → vector); a ViT has no vocabulary, so the analog is
the learned `Conv2d` patch projection. Either way it is **learned** — an `nn.Parameter`
trained by backprop — and it sits *before* the encoder. (This is the "does the embedding get
trained?" answer: yes, because it is a Parameter. Contrast the fixed sinusoidal positional
table in the transformer note, which is a *buffer* — also model state, saved and moved with
the model, but never trained.)

**(b) Output embedding — the latent representation.** The encoder's *output*: the CLS vector
`(B, D)`, or the full `(B, 197, D)` patch tokens. Each vector now encodes global context from
the whole image. This is the model's *product* — a point in a latent space you can compare,
retrieve, cluster, or classify. Also called "features," "representation," "hidden state," or
simply "the embedding."

| | Input embedding | Output embedding |
|---|---|---|
| ViT | patch projection (`Conv2d`) + CLS + pos | encoder output: CLS `(B, D)` or patch tokens |
| Text | `nn.Embedding` lookup (ID → vector) | contextual hidden states `(B, T, D)` |
| Where | before the encoder | after the encoder |
| Learned? | yes (`nn.Parameter`) | it *is* the computed output, not a stored weight |
| Role | make tokens | the latent representation you actually use |

**DINO and CLIP mean (b).** Their "embeddings" are the *output* latent, never the input
projection:

- **CLIP** — a ViT image encoder and a text encoder each produce a pooled output vector,
  projected into a *shared* latent space and trained with a **contrastive (InfoNCE)** loss so
  that matching image↔text pairs land close and mismatched pairs land far apart. The "CLIP
  embedding" is that output vector. (Same loss family as the InfoNCE drill item.)
- **DINO / DINOv2** — a self-supervised ViT whose *output* CLS / patch tokens are used
  directly as features for kNN, linear probing, and segmentation. No labels, no classification
  head — the latent representation itself is the deliverable.

**One line:** the input embedding *makes* the tokens; the output embedding is what the model
learned to *say* about them — and it is the output that retrieval, CLIP, and DINO consume.

---

## 6. The takeaway: attention is attention

ViT proves that the same self-attention block that resolves "The animal... it" in text
also reads patch-to-patch relationships in images — the *same* encoder architecture, with
only the front end (pixels → tokens) and back end (CLS → logits) swapped. Other domains
reuse the identical primitive over different token types — a time-series model attends
over timesteps, a graph transformer attends over nodes.

**One sentence to remember:** *patches, timesteps, and nodes are all just tokens fed into
the same attention mechanism.*

And the tradeoff that comes with it, worth saying in the same breath: attention gives up
the built-in priors of convolutions (locality, weight sharing), so it is more flexible but
more data-hungry — the same reason a graph transformer needs more data than a fixed-
neighborhood GCN.

---

[← Transformer Architecture](transformers.html) · [Back to contents →](index.html)
