---
title: Broadcasting in PyTorch
---

# Broadcasting in PyTorch

Broadcasting is how PyTorch lets you combine tensors of *different* shapes in one
elementwise operation, without writing loops or copying data. It is also where cold
implementations quietly break: almost every shape bug in the [Transformer](transformers.html)
and [ViT](vision-transformers.html) notes — the softmax `keepdim`, the positional-encoding
add, the attention mask — is a broadcasting decision. Get the rules automatic and those
bugs stop happening.

---

## 1. What problem it solves

You constantly need to combine a big tensor with a smaller one: add a per-feature bias to
every token, subtract each row's max from that row, add one positional table to a whole
batch. The naive way is to *copy* the small tensor until the shapes match, then add. That
wastes memory and is tedious. **Broadcasting does that stretch virtually** — no copy — and
applies the op as if the shapes matched.

```python
x = torch.randn(2, 5, 64)   # (B, T, D) — a batch of token vectors
b = torch.randn(64)         # (D,)       — one bias per feature

y = x + b                   # works: b is stretched across B and T, no copy
print(y.shape)              # torch.Size([2, 5, 64])
```

`b` was never physically expanded to `(2, 5, 64)`; PyTorch just read the one row of 64
values repeatedly. That is the whole idea. The rest is *when* it is allowed.

---

## 2. The two rules

Compare shapes **right-aligned** (from the last dimension backward). For each aligned pair
of dimensions, the op is legal if **either**:

1. the two sizes are **equal**, or
2. one of them is **1** (that axis gets stretched), or
3. one tensor **runs out of dimensions** on the left (treated as size 1 → stretched).

The result takes the **larger** size in every position.

```
   x:  (2, 5, 64)
   b:      ( 64)      ← fewer dims: left-pad with 1 → (1, 1, 64)
        ----------
        aligned pairs:  64 vs 64 ✓   5 vs 1 ✓ (stretch)   2 vs 1 ✓ (stretch)
   →   (2, 5, 64)
```

If any aligned pair is neither equal nor has a 1, it is an error:

```python
torch.randn(2, 5, 64) + torch.randn(5)   # 64 vs 5 → RuntimeError: not broadcastable
```

A quick table of the patterns you actually meet:

| A | B | Result | Why |
|---|---|---|---|
| `(B, T, D)` | `(D,)` | `(B, T, D)` | per-feature bias, stretched over B, T |
| `(B, T, D)` | `(T, D)` | `(B, T, D)` | positional table, stretched over B |
| `(B, T, T)` | `(B, T, 1)` | `(B, T, T)` | per-row max/sum, stretched over columns |
| `(1, 1, D)` | `(B, N, D)` | `(B, N, D)` | the CLS token expanded to a batch |
| `(N, 1)` | `(M,)` → `(1, M)` | `(N, M)` | **outer product** — often an accident (see §5) |

---

## 3. The examples from your own notes

Everything you have already read is broadcasting. Naming it makes the shapes obvious.

**Positional encoding add** (transformer note, Step 2): a `(T, D)` table added to a
`(B, T, D)` batch. Rule 3 left-pads the table to `(1, T, D)`, then rule 2 stretches the
leading 1 across the batch.

```python
x = token_emb + pos_enc     # (B, T, D) + (T, D) → (B, T, D)
```

**Softmax stability** (transformer note, Step 3): why `keepdim=True` is not optional.

```python
scores = torch.randn(2, 5, 5)          # (B, T, T)
m = scores.max(dim=-1, keepdim=True).values   # (B, T, 1)  ← keep the reduced axis as 1
scores - m                              # (B, T, T) - (B, T, 1) → (B, T, T) ✓ per-row
```

Without `keepdim`, `max` returns `(B, T)`. Then `(B, T, T) - (B, T)` right-aligns as
`T vs T` (ok) and `T vs B` — which is **silently wrong** if `T == B`, or an error
otherwise. The trailing `1` is what forces each element to subtract *its own row's* max.

**Attention mask** (transformer note, Step 4): a `(1, 1, T, T)` or `(T, T)` mask stretched
across batch and heads before softmax.

```python
scores = scores.masked_fill(mask == 0, float('-inf'))   # mask broadcasts over B, h
```

**CLS token** (ViT note, Step 2): a single `(1, 1, D)` parameter stretched to `(B, 1, D)`.

```python
cls = self.cls.expand(B, -1, -1)        # (1, 1, D) → (B, 1, D)
```

---

## 4. Making broadcasting happen on purpose: `unsqueeze` / `None`

Often two tensors *almost* line up and you need to insert a size-1 axis so the stretch
lands on the right dimension. `unsqueeze(k)` (or indexing with `None`) inserts a length-1
axis at position `k`.

```python
pos = torch.arange(5)          # (5,)
pos.unsqueeze(1).shape         # (5, 1)     ← same as pos[:, None]
pos.unsqueeze(0).shape         # (1, 5)     ← same as pos[None, :]

# The positional-encoding note used exactly this:
pos  = torch.arange(T).unsqueeze(1)     # (T, 1)
freq = torch.arange(0, D, 2)            # (D/2,)
table = pos * freq                      # (T, 1) * (D/2,) → (T, D/2)  outer product
```

`(T, 1) * (D/2,)`: right-align → `1 vs D/2` (stretch to D/2) and `T vs (missing)` (stretch
to T). Result `(T, D/2)` — every position times every frequency. This is the deliberate,
useful version of the outer-product pattern.

---

## 5. The two bugs that cost you time

**Bug 1 — the missing `keepdim` (silent).** Reducing without `keepdim` drops the axis, and
the later broadcast either errors or, worse, aligns against the *wrong* axis and runs. Rule
of thumb: **any reduction (`sum`, `max`, `mean`) that you will broadcast back against the
original tensor needs `keepdim=True`.**

**Bug 2 — the accidental outer product.** Subtracting a `(N,)` from a `(N, 1)` gives an
`(N, N)` matrix, not an `(N,)` vector:

```python
a = torch.randn(4, 1)     # (4, 1)
b = torch.randn(4)        # (4,) → (1, 4)
(a - b).shape             # (4, 4)  ← probably NOT what you wanted
```

This shows up when you forget to squeeze a `(N, 1)` back to `(N,)`, or mix up which tensor
should carry the trailing 1. When a result is unexpectedly 2D (or huge), suspect this.

**Debugging habit:** when a shape error or a wrong-sized result appears, print the two
shapes and right-align them on paper. The offending pair is always visible in three seconds.

---

## 6. `expand` vs `repeat` (broadcasting vs. copying)

Sometimes you want a real, materialized tensor of the broadcast shape. Two ways, and they
differ in memory:

| Op | Copies data? | Use when |
|---|---|---|
| `x.expand(shape)` | **no** — sets stride 0, a view | you only read the stretched tensor (cheap) |
| `x.repeat(counts)` | **yes** — allocates | you need to write into it independently |

The ViT CLS token used `expand` (read-only, zero-copy). Reach for `repeat` only when you
actually need distinct, writable copies.

---

## 7. Smoke test

```python
import torch

# 1. per-feature bias over a batch
x = torch.randn(2, 5, 64)
b = torch.randn(64)
assert (x + b).shape == (2, 5, 64)

# 2. per-row max subtraction with keepdim
s = torch.randn(2, 5, 5)
m = s.max(dim=-1, keepdim=True).values          # (2, 5, 1)
assert m.shape == (2, 5, 1)
assert (s - m).shape == (2, 5, 5)

# 3. outer product on purpose (positional table)
pos  = torch.arange(5).unsqueeze(1)             # (5, 1)
freq = torch.arange(0, 64, 2)                   # (32,)
assert (pos * freq).shape == (5, 32)

# 4. expand is zero-copy (shares storage with the source)
cls = torch.zeros(1, 1, 64)
big = cls.expand(2, 1, 64)
assert big.shape == (2, 1, 64)
assert big.data_ptr() == cls.data_ptr()   # same memory → expand made no copy

# 5. a non-broadcastable pair raises
try:
    _ = torch.randn(2, 5, 64) + torch.randn(5)
    raise AssertionError("should have failed")
except RuntimeError:
    pass

print("SMOKE TEST PASSED")
```

---

## 8. The rules in one line

**Right-align the shapes; every aligned pair must be equal or have a 1; missing dims are
1s; the result takes the bigger size.** Keep `keepdim=True` on any reduction you will
broadcast back, and suspect an accidental outer product whenever a result is unexpectedly
2D.

---

[← Back to contents](index.html) · [Transformer Architecture →](transformers.html)
