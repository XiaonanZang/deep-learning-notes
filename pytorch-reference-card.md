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

[← Back to contents](https://xiaonanzang.github.io/deep-learning-notes/index.html)
