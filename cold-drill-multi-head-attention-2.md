---
title: "Cold Drill 2: Multi-Head Attention (Spaced Repetition)"
---

# Cold Drill 2: Multi-Head Attention (Spaced Repetition)

**What this drill is about:** the *same* problem as
[Cold Drill 1](https://xiaonanzang.github.io/deep-learning-notes/cold-drill-multi-head-attention.html)
— multi-head self-attention + a stable softmax from scratch — re-attempted cold from a blank
file a couple of days later. That gap is the whole point: **spaced repetition** is what turns a
mechanism you *understood once* into one you can *type correctly under time pressure*.

This page is the second pass, in the same four parts: **the problem**, a **realistic second
attempt** (reproduced as written, bugs and all), a **critique** — including what improved and
what regressed versus the first pass — and a **corrected, runnable solution**.

The interesting result: the *design* errors from pass 1 were fixed, but the code **still did not
run**, for entirely new reasons. The failure moved. Watching it move is more instructive than any
single clean solution.

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

**Rules:** blank file, from scratch, make it run — this pass targets **under 20 minutes** (the
first pass took ~27).

---

## 2. A realistic second attempt

An actual cold, ~20-minute second attempt, reproduced exactly as written. It does **not** run.
Read it against the spec before the critique — the bugs here are different from pass 1.

```python
# Skip the reasoning now, since I already did it in the first pass.
# Directly write the softmax function, and the class for the multi-attention head.
import torch
import torch.nn as nn
import math

def Soft_Max(x, dim=-1):
  # Calculate the max per each row while maintaining the correct shape and format
  x_max = x.max(dim, keepdim==true).value()
  e = torch.exp(x-x_max)
  # We minus max to keep stable, while make sure the shape is correct
  return (x-x_max)/e.sum(dim, keepdim==true)

class Multi_Attn_Head(nn.modules):
  # We need a proper init function (include all the states) and the forward function (behavior)
  def __init__(self, d_model):
    super().__init__()  # Make sure this class and its state is properly registered
    # We need four weights: Q, K, V, and output weights.
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_o = nn.Linear(d_model, d_model)

  # x is projected with q, k, v weights, then dot-product, scale, multiply V, then output weights.
  def forward(self, x, n_heads):
    assert (d_model // n_head)          # meant as a validity check
    d_k = d_model // n_head
    B, T, D = x.shape()
    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)
    Q_k = Q.view(B,T,n_heads,d_k).transpose(1,2)
    K_k = K.view(B,T,n_heads,d_k).transpose(1,2)
    V_k = V.view(B,T,n_heads,d_k).transpose(1,2)
    A_k = Q_k @ K_k.transpose(-1,-2) / sqrt(d_k)
    A = (A_k*V_k).contiguous.transpose(1,2).view(B,T,D)
    return W_o(A), A
```

---

## 3. The critique

### What improved versus pass 1 (real progress)

Every one of the *design* bugs that sank the first attempt is now fixed:

- **It is an `nn.Module`** with `__init__`, `super().__init__()`, and all four learnable
  projections `W_q / W_k / W_v / W_o` — pass 1 had plain functions and no projections at all.
- **Head split is `transpose(1, 2)`** — pass 1 used the wrong pair `transpose(-1, -2)`.
- **The scores are scaled by `/ √d_k`** — missing in pass 1.
- **V is projected and used as the value**, and there is a **head-merge** back to `(B, T, D)` —
  both absent in pass 1.

The architecture is now in muscle memory. So where did it fail?

### Conceptual bugs — new this pass

| # | Bug | Why it is wrong | Fix |
|---|-----|-----------------|-----|
| **1** | **Softmax is never called in `forward`.** `A_k` (raw scores) goes straight into the value step. | Attention *is* the softmax-weighted average of values. Without it you are just multiplying raw dot-products by V. Pass 1 actually had this right — it regressed. | `A = Soft_Max(scores, dim=-1)` before the value matmul |
| **2** | **Softmax numerator is wrong:** returns `(x - x_max) / Σe`. | `e = exp(x - x_max)` was computed and then not used in the numerator. Softmax is `e / Σe`, not `(x - x_max) / Σe`. | `return e / e.sum(dim, keepdim=True)` |
| **3** | **`A_k * V_k` is elementwise.** | Attention aggregates values by a **matmul** over positions, not a Hadamard product. (It also shape-mismatches: `(B,h,T,T)` vs `(B,h,T,d_k)`.) The value operand is right this time — the operator is wrong. | `out = A @ V_k` |

> The value step is the instructive one across both passes. Pass 1 wrote `A @ x + V` (right
> operator, wrong operand + a phantom residual). Pass 2 wrote `A * V` (right operand, wrong
> operator). Two different ways to get `softmax(...) @ V` not-quite-right — a good sign of what to
> drill next.

### Syntax slips (intent clear, written under the clock — but they crash the run)

- `nn.modules` → **`nn.Module`** — this is what actually stopped the run, right at the class line
- `keepdim==true` → `keepdim=True` (twice); `.value()` → `.values`
- `x.shape()` → `x.shape` (property, not a call)
- `W_q(x)` → `self.W_q(x)` (and `W_k`, `W_v`, `W_o`) — the projections live on `self`
- `sqrt` → `math.sqrt`; `.contiguous` → `.contiguous()` (a method call)
- `d_model` / `n_head` are out of scope and mis-named → store `self.d_model`, `self.n_heads`
- `assert (d_model // n_head)` is not a validity check → `assert d_model % n_heads == 0`

### Scorecard

| Dimension | Pass 1 | Pass 2 | Note |
|-----------|:---:|:---:|------|
| Understanding | strong | strong | `nn.Module` framing gained; but the softmax call was dropped from the pipeline |
| Design | right spine, big gaps | **much stronger** | projections, `transpose(1,2)`, scaling, head-merge all now present |
| Correctness | does not run | **does not run** | the *must-run gate* failed both times, for different reasons |
| Communication | strong | lighter | reasoning deliberately skipped ("did it in pass 1") — fine for a speed drill |

**The takeaway across two passes:** design improved a lot, but the answer still did not run. When
the *same must-run gate* fails twice for *different* reasons, the message is clear — the gap is no
longer understanding the architecture, it is **producing code that runs**. The highest-value fix
is to drill the next pass **in a real REPL/IDE**, so `==` vs `=`, `.values`, `self.`, and "did I
actually call softmax" surface as errors in the moment instead of in review.

---

## 4. The corrected solution

Runnable, verified on CPU with PyTorch 2.x — the softmax matches `F.softmax` and the module
returns `(2, 5, 32)`. (Same target as pass 1's solution; every fix from §3 applied.)

```python
import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    x_max = x.max(dim, keepdim=True).values      # .values — max returns (values, indices)
    e = torch.exp(x - x_max)                      # stability: shift by the row max
    return e / e.sum(dim, keepdim=True)           # numerator is e, normalize along dim only


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, t, B, T):
        # (B, T, D) -> (B, T, h, d_k) -> (B, h, T, d_k).  transpose(1, 2), not (-1, -2).
        return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B, T, D = x.shape                                       # property, not a call
        Q = self._split_heads(self.W_q(x), B, T)                # self.W_q, projections on self
        K = self._split_heads(self.W_k(x), B, T)
        V = self._split_heads(self.W_v(x), B, T)

        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B, h, T, T), SCALED
        attn = softmax(scores, dim=-1)                            # <-- the step pass 2 forgot
        out = attn @ V                                           # (B, h, T, d_k) — MATMUL, not *

        out = out.transpose(1, 2).contiguous().view(B, T, D)     # .contiguous() is a call
        return self.W_o(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    y = mha(x)
    print("output shape:", tuple(y.shape))
    assert y.shape == (2, 5, 32), y.shape
    print("OK")
```

---

## Takeaways

1. **A softmax that is never called is the easiest step to drop** once the surrounding code gets
   longer. The pipeline is `scores → softmax → @V`; do not skip the middle under the clock.
2. **`e / Σe`, not `(x - x_max) / Σe`.** Compute the exponential, then use it in the numerator.
3. **Values are aggregated with `@`, not `*`.** Getting the operand right (V, not x) is only half
   of `softmax(...) @ V`.
4. **The failure moves as you improve.** Pass 1 failed on design; pass 2 failed on execution
   precision. Track *where* it fails, not just *that* it fails — that is what tells you what to
   drill next.
5. **When the must-run gate keeps failing, change the environment, not the effort.** Drilling in a
   real REPL surfaces `==`, `.values`, and `self.` as live errors — the exact class of bug that a
   blank-page attempt hides until review.

*Read it, close the page, and type it from blank. The gap between the two — measured across
repeated passes — is the drill.*
