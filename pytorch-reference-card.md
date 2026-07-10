---
title: PyTorch Reference Card
---

# PyTorch Reference Card

A one-glance reference for the mechanical parts that are easy to *understand* and easy to
*fumble under a clock*: the data plumbing, the training loop, and the tensor-axis operations
that cause most "it doesn't run" moments. Understanding a mechanism and typing it correctly
under time pressure are different skills — this card targets the second.

---

## A. Custom Dataset & DataLoader

The three methods a `Dataset` must define, and how the `DataLoader` batches it:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)   # (N, D)
        self.y = torch.as_tensor(y, dtype=torch.long)      # (N,)
    def __len__(self):
        return len(self.X)                                 # number of samples
    def __getitem__(self, i):
        return self.X[i], self.y[i]                        # ONE sample, as tensors

ds = MyDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
loader = DataLoader(ds, batch_size=16, shuffle=True)       # batches + shuffles for you
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)                                  # (16, 4) (16,)
```

- `__len__` → dataset size (the `DataLoader` needs it to form batches).
- `__getitem__(i)` → **one** sample as tensors; the `DataLoader` stacks `batch_size` of them.
- `shuffle=True` for training, `False` for eval. Non-uniform samples (variable length, etc.)
  → pass your own `collate_fn=...`.

---

## B. The training loop

The six verbs, in order — the fully annotated version lives in
[Activations, Losses, Optimizers §5](https://xiaonanzang.github.io/deep-learning-notes/activations-losses-optimizers.html):

```
opt.zero_grad()        # 1. clear old grads      (skip this → grads accumulate)
logits = model(xb)     # 2. forward
loss   = criterion(logits, yb)   # 3. loss       (CE wants raw logits, no softmax)
loss.backward()        # 4. autograd → grads
opt.step()             # 5. update the weights
sched.step()           # 6. advance the LR       (only if scheduling)
```

`model.train()` / `model.eval()` toggle dropout & BatchNorm; wrap evaluation in
`@torch.no_grad()` (or `with torch.no_grad():`) to skip grad tracking.

---

## C. Axis-semantics cheat-sheet (the "it doesn't run" antidote)

The operations that cause most cold-drill run failures, one line each. (For the *why*, see
[Broadcasting](https://xiaonanzang.github.io/deep-learning-notes/broadcasting.html).)

| Need | Write | Slip to avoid |
|---|---|---|
| softmax over an axis | `F.softmax(x, dim=-1)` | `dim` is the axis **index**, not its size; the 2nd positional arg *is* `dim` |
| add a new axis | `x[:, None, :]` | not `x[:None:]` — `None` goes in a **comma-separated** position |
| causal / graph mask | `scores.masked_fill(mask, float('-inf'))` **before** softmax | mask the *logits* with `-inf`, not the weights with `0` |
| all-pairs (outer) sum | `a[:, None] + b[None, :]` → `(N, N)` | keep operand order consistent with your softmax `dim` |
| reshape | `x.reshape(...)` (`view` for a no-copy guarantee) | `view` needs contiguous memory; `reshape` is always safe |
| swap axes | `x.transpose(-2, -1)` / `x.permute(0, 2, 1)` | `transpose` = exactly 2 axes; `permute` = full reorder |
| batched dot / contraction | `(A * B).sum(-1)` or `torch.einsum('ijh,jho->iho', A, B)` | in einsum, the index **missing from the output is the summed one** (`j` here) |

---

## D. `nn.Parameter` & initialization

- A learnable tensor must be wrapped in `nn.Parameter(...)` (or registered as a buffer if
  non-learnable) to move with `.to(device)` and reach the optimizer. See
  [Module Pattern](https://xiaonanzang.github.io/deep-learning-notes/pytorch-module-pattern.html).
- **Multiplicative** weights (multiplied into the signal — `W`, attention vectors) → random
  init (`nn.init.xavier_uniform_`) to break symmetry and preserve variance.
- **Additive** parameters (bias, positional embedding, class token) → **zeros are fine**; the
  gradient flows straight through and there is no symmetry to break.
- `torch.empty(...)` holds uninitialized **garbage** — always `nn.init.*_` it. `nn.Linear`
  already initializes its `.weight` for you.

---

## E. U-Net — composing the primitives (segmentation)

Segmentation = **per-pixel classification**: the output is `(B, num_classes, H, W)`, one class
distribution per pixel. U-Net is the canonical shape — an **encoder that downsamples**, a
**decoder that upsamples back**, and **skip connections** that hand the encoder's high-resolution
detail to the decoder so the boundaries stay sharp. It is a clean exercise in wiring the earlier
primitives together (`Conv2d` down, `ConvTranspose2d` up, `ModuleList`, channel-axis `cat`).

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(conv → BN → ReLU) × 2 — the repeating unit of every U-Net stage."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, feats=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2)                       # halves H,W between encoder stages

        c = in_ch                                          # encoder: DoubleConv per level
        for f in feats:
            self.downs.append(DoubleConv(c, f)); c = f

        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2)

        for f in reversed(feats):                          # decoder: up-conv, then fuse the skip
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))   # 2× upsample
            self.ups.append(DoubleConv(f * 2, f))                        # in = skip(f) + up(f)

        self.head = nn.Conv2d(feats[0], num_classes, 1)    # 1×1 conv → per-pixel class logits

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)                # SAVE full-res features for the skip
            x = self.pool(x)               # then downsample

        x = self.bottleneck(x)
        skips = skips[::-1]                 # deepest skip first

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)                         # ConvTranspose up (2×)
            x = torch.cat([skips[i // 2], x], dim=1)   # concat skip on the CHANNEL axis (dim=1)
            x = self.ups[i + 1](x)                     # DoubleConv fuses skip + upsampled

        return self.head(x)                # (B, num_classes, H, W)

m = UNet(in_ch=3, num_classes=2)
y = m(torch.randn(1, 3, 256, 256))
print(tuple(y.shape))                      # (1, 2, 256, 256) — per-pixel logits  (~31M params)
```

- **Encoder** halves `H,W` and doubles channels at each stage (`MaxPool2d` between `DoubleConv`s);
  the **decoder** mirrors it, `ConvTranspose2d` doubling `H,W` back up.
- **Skip connection = `torch.cat` on the channel axis (`dim=1`)**, *not* an add — the decoder input
  is `skip(f) + upsampled(f) = 2f` channels, which is why each up-stage `DoubleConv` takes `f*2 → f`.
  This is the one shape to get right; the concat is what recovers fine detail lost to pooling.
- **`head` is a 1×1 conv** mapping the last feature map to `num_classes` — no flattening, spatial
  size preserved, so every pixel gets its own logits vector.
- **Loss** = `nn.CrossEntropyLoss()(logits, target)` with `logits (B,C,H,W)` and integer
  `target (B,H,W)` — cross-entropy over pixels (PyTorch's CE accepts the spatial dims directly).
- Same architecture is the backbone of **diffusion U-Nets** and medical/industrial segmentation.

---

[← Back to contents](https://xiaonanzang.github.io/deep-learning-notes/index.html)
