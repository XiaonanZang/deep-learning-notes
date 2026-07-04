---
title: "Cold Drill: Multi-Head Attention from Scratch"
---

# Cold Drill: Multi-Head Attention from Scratch

**What this drill is about:** implementing **multi-head self-attention** as an `nn.Module`,
together with a **numerically stable softmax**, entirely from scratch — no
`nn.MultiheadAttention`, no `F.scaled_dot_product_attention`, no `F.softmax`.

A *cold drill* is a timed, blank-file exercise: no notes, no autocomplete, ~25 minutes, and
the code has to actually run on a synthetic input. It is meant to expose the gap between
*understanding* a mechanism and being able to *type it correctly under time pressure* — a very
different skill, and the one a live coding screen actually tests.

This page is a worked example in four parts: **the problem**, a **realistic first attempt**
(reproduced as written, bugs and all), a **critique** of every issue, and a **corrected,
runnable solution**. If you want the intuition and math behind attention first, read
[Transformer Architecture](https://xiaonanzang.github.io/deep-learning-notes/transformers.html).

---

## 1. The problem

> Implement multi-head self-attention as an `nn.Module`, from scratch. Write your own softmax
> (numerically stable). It must run on a synthetic input.
> **Not allowed:** `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `F.softmax`.

**Spec**

- Input `x`: shape `(B, T, D)` — batch, sequence length, model dim.
- Config: `d_model`, `n_heads` (assume `d_model % n_heads == 0`).
- Output: shape `(B, T, D)`.
- Scaled dot-product attention **per head**: `softmax(Q Kᵀ / √d_k) · V`.
- Your own `softmax` with max-subtraction stability.
- Smoke test: instantiate, run a random `(2, 5, 32)` input, print the output shape, and
  `assert` it is `(2, 5, 32)`.

**Rules:** ~25 minutes, blank file, from scratch, make it run and self-verify on the synthetic
input.

---

## 2. A realistic first attempt

This is an actual cold, ~25-minute attempt, reproduced exactly as written. It does **not** run.
Read it before the critique and see how many issues you can spot yourself — that is the real
exercise.

```python
# Comment:
# This is how I approach this problem.
# First, let us talk about the major steps.
# 1. We need set up the proper multi head Q_k, and K_k, so that we could split properly
# 2. During each attention head we need have the Q@K^T, and then use the softmax to control the weights
# 3. After softmax the weights, we then use A@x+V to pass later on for the residual combination?

# anyway, let us write the softmax function first.
def SoftMax(x, dim=-1):
  # We already suppose the x is (B, T, T)
  # Then, we find the max along each row
  x_max = x.max(dim, keepdim==true)  # In this way, we will get a (B, T, 1) tensor
  x_soft_max = e.(x-x_max)/sum(e.(x-x_max))
  return x_soft_max

# Next, we want to set up the multi-head attention head
def Multi_Attn_Head(x, Q, K, V, n_head):
  # We gonna break the dimension into proper segments
  B, T, D = x.shape()
  D_k = D // n_head  # Obtain integer
  # We suppose to already have defined parameters: Q (B, T, D), and K (B,T,D)
  # Break Q into (B, h, T, D_k): view into (B, T, h, D_k) first, then transpose to (B, h, T, D_k)
  Q_k = Q.view(B,T,n_head,D_k).transpose(-1,-2)
  K_k = K.view(B,T,n_head,D_k).transpose(-1,-2)
  A = Q_k @ K_k.transpose(-1,-2)  # A is in a (B, h, T, T) shape
  A_soft_max = SoftMax(A)
  x = A_soft_max@x + V
  return x, A
```

---

## 3. The critique

### What is already right

The **skeleton is correct** — all four moves of attention are present and in the right order:
split into heads, `Q Kᵀ`, softmax the scores, then weight the values. Specifically good:

- The **3-step plan** up front — the decomposition is right.
- **Softmax max-subtraction** for numerical stability — correct and unprompted.
- The **view → transpose** *strategy* for head-splitting — the right idea (reshape into
  `(B, T, h, d_k)` first, then move the head axis forward).
- **Pipeline order** `Q Kᵀ → softmax → (· V)` — correct.
- `D_k = D // n_head` integer split — correct.

The failures below are all about *execution precision*, not about misunderstanding attention.

### Conceptual bugs (these are the ones that matter)

| # | Bug | Why it is wrong | Fix |
|---|-----|-----------------|-----|
| **A** | Softmax denominator has no reduction axis — `sum(e.(...))` sums the **whole tensor** | Softmax must normalize **along one axis** so each row sums to 1; a global sum mixes unrelated rows | `e.sum(dim, keepdim=True)` |
| **B** | Head-split uses `.transpose(-1, -2)` | That swaps the **last two** axes: `(B,T,h,d_k) → (B,T,d_k,h)`, contradicting the comment's own target `(B,h,T,d_k)`. It silently corrupts every downstream shape. | `.transpose(1, 2)` |
| **C** | No `/ √d_k` scaling on the scores | Un-scaled dot products grow with `d_k`, pushing softmax into saturated regions with vanishing gradients — the whole point of *scaled* dot-product attention | `scores / math.sqrt(d_k)` |
| **D** | `A_soft_max @ x + V` fuses three different things | It should be `A_soft_max @ V` (the per-head **values**), not `@ x` (raw input). And `+ V` treats V as a **residual** — but V is the matrix you matmul *with*. The residual `x + attn_out` lives in the **encoder layer**, outside attention. | `out = A_soft_max @ V_k` |
| **E** | Heads are never merged back | After weighting, the shape is `(B, h, T, d_k)`; the contract is `(B, T, D)` | `.transpose(1,2).contiguous().view(B,T,D)` |
| **F** | Not an `nn.Module`; no smoke test | The spec asked for a module with learnable `W_q/W_k/W_v` (and usually `W_o`) projecting from `x`, plus a runnable check — both were part of "done" | see §4 |

> **Bug B is the instructive one.** The comment states the correct target shape `(B,h,T,d_k)`,
> but the code reaches for `-1, -2` out of habit. Knowing the target and typing the right axis
> are two separate skills — which is exactly what a cold drill trains.

### Syntax slips (intent was clear, written under the clock)

These would stop it compiling, but they are shorthand, not knowledge gaps:

- `keepdim==true` → `keepdim=True` (and `x.max(dim, keepdim=True)` returns a named tuple, so
  you need `.values`)
- `e.(x-x_max)` → `torch.exp(x - x_max)`
- `sum(...)` → `torch.sum(...)` / `.sum(...)` (see bug **A** for the axis)
- `x.shape()` → `x.shape` (it is a property, not a call)
- missing `import torch` / `import math`

### Scorecard

| Dimension | Assessment |
|-----------|------------|
| Understanding | Strong — clear plan and shape story |
| Design | Right spine; missing scaling, head-merge, projections; wrong value-matmul |
| Correctness | Does not run — syntax slips plus the axis/normalization bugs |
| Communication | Strong — reasoning narrated throughout |

The concept is there; the fingers are not yet. That is the normal state after a *first* cold
pass, and precisely why the drill is worth repeating from a blank file on a later day.

---

## 4. The corrected solution

Runnable, verified on CPU with PyTorch 2.x. Every fix from §3 is applied and commented.

```python
import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    # Numerically stable: subtract the per-row max before exponentiating.
    x_max = x.max(dim, keepdim=True).values      # (..., 1); .values because max returns (values, indices)
    e = torch.exp(x - x_max)                      # shifting by a constant leaves softmax unchanged
    return e / e.sum(dim, keepdim=True)           # normalize ALONG dim only, keepdim so it broadcasts back


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Learnable projections: one Linear each for Q, K, V, plus an output projection.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, t, B, T):
        # (B, T, D) -> (B, T, h, d_k) -> (B, h, T, d_k).  NOTE: transpose(1, 2), NOT (-1, -2).
        return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B, T, D = x.shape                                    # property, not a call
        Q = self._split_heads(self.W_q(x), B, T)             # (B, h, T, d_k)
        K = self._split_heads(self.W_k(x), B, T)
        V = self._split_heads(self.W_v(x), B, T)

        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)   # (B, h, T, T), SCALED
        attn = softmax(scores, dim=-1)                            # weights over the key axis
        out = attn @ V                                            # (B, h, T, d_k) — matmul with VALUES

        out = out.transpose(1, 2).contiguous().view(B, T, D)      # merge heads back to (B, T, D)
        return self.W_o(out)                                      # output projection


if __name__ == "__main__":
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    y = mha(x)
    print("output shape:", tuple(y.shape))
    assert y.shape == (2, 5, 32), y.shape
    print("OK")
```

Running it prints:

```
output shape: (2, 5, 32)
OK
```

### The shape story (memorize this — it is the whole exercise)

```
x            (B, T, D)
Q,K,V = x @ W_{q,k,v}                       each (B, T, D)     # nn.Linear projections
split heads: view (B,T,h,d_k) → transpose(1,2) → (B, h, T, d_k)   # transpose(1,2), not (-1,-2)
scores = Q @ Kᵀ / √d_k                       (B, h, T, T)
attn   = softmax(scores, dim=-1)             (B, h, T, T)      # sum over last axis, keepdim
out    = attn @ V                            (B, h, T, d_k)
merge  : transpose(1,2).contiguous().view    (B, T, D)
out    = out @ W_o                           (B, T, D)
# the residual (x + out) and LayerNorm belong to the ENCODER LAYER, not to attention
```

---

## Takeaways

1. **Softmax normalizes along one axis.** `keepdim=True` on both the max and the sum is what
   lets the result broadcast back cleanly. A missing axis is a silent, wrong-answer bug.
2. **`transpose(1, 2)` splits heads, not `transpose(-1, -2)`.** Knowing the target shape is not
   the same as typing the right axis. This is the single most common cold-drill slip.
3. **Attention returns `softmax(...) @ V`.** The residual connection and normalization are the
   encoder layer's responsibility, one level up — do not fuse them into the attention block.
4. **Scale by `√d_k`.** It is in the name for a reason.
5. **"Runs on a synthetic input" is part of the answer.** A smoke test that instantiates the
   module and checks the output shape is not optional polish; it is how you catch A–F in the
   last two minutes.

*Read it, then close the page and type it from blank. The gap between the two is the drill.*
