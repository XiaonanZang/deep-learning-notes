---
title: Activations, Losses, Optimizers, and Schedulers
---

# Activations, Losses, Optimizers, and Schedulers

A briefing on the four training-loop ingredients that recur across every architecture,
with from-scratch code for the ones you might have to *write* and idiomatic built-ins for
the ones you would just *call*. This note is deliberately practical: the
[Transformer](transformers.html) and [ViT](vision-transformers.html) notes assume all of
this, and a 30-minute coding test can hand you a loss to implement or ask you to reason
about a training curve.

**What is worth typing cold vs. what is worth explaining:**

| Ingredient | Likely to *write* from scratch? | Priority |
|---|---|---|
| Activations (GELU, softmax, ReLU) | Sometimes — softmax yes, others are one-liners | Know cold |
| Losses (cross-entropy, MSE, **InfoNCE**) | **Yes** — a contrastive/CE loss is a plausible problem | Know cold |
| Optimizers (SGD, Adam) | Rarely — you call `torch.optim` | Explain, don't drill |
| Schedulers (warmup, cosine) | Rarely | Explain, don't drill |

---

## 1. Activations — why any nonlinearity at all

Stacking linear layers with nothing between them is pointless: two matrix multiplies
compose into a single matrix multiply, so a 10-layer linear net has exactly the expressive
power of one layer. **The nonlinearity between linears is what gives depth its power** —
this is the reason the FFN in the transformer note is `Linear → GELU → Linear` and not two
back-to-back linears.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 8)

# ReLU: max(0, x). Cheap, sparse, the classic default. Dead for x < 0.
F.relu(x)

# GELU: smooth, ~x * sigmoid(1.7x). Standard in transformers (used in the FFN note).
F.gelu(x)

# Sigmoid: squashes to (0, 1). Use for a single binary probability / gates.
torch.sigmoid(x)

# Tanh: squashes to (-1, 1). Zero-centered; older RNN default.
torch.tanh(x)
```

**Softmax** is the one activation you should be able to write from scratch, because it is
*inside* attention and every classifier. It turns a vector of scores into a probability
distribution. The from-scratch, numerically-stable version (transformer note, Step 3):

```python
def softmax(x, dim=-1):
    # subtract the row max before exp so exp never overflows; mathematically identical.
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)
```

Rule of thumb: **ReLU/GELU between hidden linears; softmax for a categorical output or
attention weights; sigmoid for a single independent probability.**

### From scratch

Most of these are one or two lines — which is exactly why an interviewer might hand you one
to check you *understand* it rather than reach for `F.`. All verified to match the built-in.

```python
def relu(x):
    return x.clamp(min=0)                 # max(0, x), elementwise

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))        # squash to (0, 1)

def tanh(x):
    return 2 * sigmoid(2 * x) - 1         # zero-centred; = 2σ(2x) − 1

def gelu(x):                              # exact GELU
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

def gelu_tanh(x):                         # the tanh approximation many impls ship
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
```

**Softmax** is the one to be careful with — it needs the max-subtraction for stability, and
it is already written from scratch above (the version inside attention). The others are safe
to type naively; the only real gotcha is **naive `sigmoid` can overflow for large negative
`x`** (`exp(-x)` explodes), which is why `logsigmoid` / `bce_with_logits` exist for the
loss path. For a plain forward activation the naive form is fine.

---

## 2. Losses — the number you differentiate

The loss turns predictions + targets into one scalar to minimize. Three you should know,
plus one from the interviewer's world.

### Cross-entropy (classification)

The default for "pick one of C classes." **`nn.CrossEntropyLoss` expects raw logits, not
probabilities** — it applies `log_softmax` internally. Feeding it softmax outputs is a
classic double-softmax bug.

```python
logits  = torch.randn(8, 10)              # (B, C) — raw scores, NO softmax
targets = torch.randint(0, 10, (8,))      # (B,)   — integer class indices
loss = F.cross_entropy(logits, targets)   # scalar
```

### MSE (regression)

Mean squared error for continuous targets.

```python
pred = torch.randn(8, 1)
y    = torch.randn(8, 1)
loss = F.mse_loss(pred, y)
```

### BCE (multi-label / binary)

Binary cross-entropy for independent yes/no outputs. Use the `with_logits` version for
stability (it fuses the sigmoid).

```python
logits = torch.randn(8, 3)                # 3 independent binary labels
y      = torch.randint(0, 2, (8, 3)).float()
loss = F.binary_cross_entropy_with_logits(logits, y)
```

### InfoNCE / contrastive (the interviewer's turf — worth writing cold)

Pulls matched pairs together and pushes mismatched pairs apart in embedding space. It is a
**cross-entropy over cosine similarities**: for each anchor, the positive should score
highest among all candidates. This is the shape of a plausible 30-minute problem for a
retrieval / RLHF interviewer, so here it is from scratch:

```python
def info_nce(z1, z2, temperature=0.07):
    # z1, z2: (N, D) — matched pairs. Row i of z1 matches row i of z2.
    z1 = F.normalize(z1, dim=-1)              # unit vectors → dot product = cosine sim
    z2 = F.normalize(z2, dim=-1)

    # Similarity matrix: logits[i, j] = cosine(z1_i, z2_j). (N, N)
    logits = z1 @ z2.T / temperature          # temperature sharpens the distribution

    # The correct match for anchor i is column i, so targets are 0, 1, ..., N-1.
    targets = torch.arange(z1.shape[0])

    # Cross-entropy treats each row as a classification over the N candidates,
    # with the diagonal as the positive class.
    return F.cross_entropy(logits, targets)
```

The trick to remember: **contrastive learning is just cross-entropy where the "classes"
are the other items in the batch and the label is the diagonal.**

---

## 2b. Losses from scratch — the cold-drill target

The versions above call `F.*` built-ins. In a live coding test you may have to *write the
math itself*, with no built-in to lean on. These are the "type this cold" targets: each is
runnable and verified to match the PyTorch built-in. Drill them from a blank file, then
check against these.

### Cross-entropy from scratch

CE = `-log p[target]`, where `p = softmax(logits)`. The whole game is doing it in
**log-space** so you never compute `log(softmax(...))` directly (that underflows).

```python
def cross_entropy(logits, targets):
    # logits (B, C) raw scores; targets (B,) integer class indices
    logp = logits - logits.logsumexp(dim=-1, keepdim=True)   # = log_softmax, stable
    return -logp[torch.arange(len(targets)), targets].mean() # pick the target's log-prob
```

**Gotcha:** work in log-space via `logsumexp`; never `torch.log(softmax(x))`. The
`arange`-plus-`targets` fancy-index is how you pull one entry per row.

### MSE from scratch

Mean of the squared residuals. Nothing subtle — but say "mean over *all* elements" out loud.

```python
def mse_loss(pred, y):
    return ((pred - y) ** 2).mean()
```

**Gotcha:** `.mean()` averages over every element (batch × features). Use `.sum()` only if
you specifically want unaveraged loss.

### BCE (with logits) from scratch

Binary cross-entropy fused with the sigmoid, in its numerically stable form. The naive
`-(y*log(σ(z)) + (1-y)*log(1-σ(z)))` overflows for large `|z|`; the identity below never does.

```python
def bce_with_logits(z, y):
    # z: raw logits, y: 0/1 targets, same shape
    # stable form of -[y·logσ(z) + (1-y)·logσ(-z)]
    return (z.clamp(min=0) - z * y + torch.log1p(torch.exp(-z.abs()))).mean()
```

**Gotcha:** take logits, not probabilities — the `clamp`/`abs` trick is what makes it stable,
which is exactly why `binary_cross_entropy_with_logits` exists over `sigmoid` + `BCELoss`.

### InfoNCE from scratch

Already written from scratch in §2 above (it is the one contrastive loss worth drilling for
this interviewer). It is the fourth cold-drill target — **cross-entropy over a cosine-similarity
matrix, label = the diagonal.** Kept in one place to avoid a second copy.

**One-line recall for all four:** CE = pick the target's log-prob in log-space · MSE = mean
squared residual · BCE = stable logit form · InfoNCE = CE over similarities, label = diagonal.

---

## 3. Optimizers — how the weights actually move (explain, do not drill)

After `loss.backward()` fills `.grad` on every parameter, the optimizer uses those
gradients to update the weights. You almost never hand-write this in an interview — you
call `torch.optim` — but you should be able to explain the progression:

| Optimizer | One-line idea |
|---|---|
| **SGD** | `w -= lr * grad`. Simple; needs a good LR and often momentum. |
| **SGD + momentum** | accumulate a velocity of past gradients → smoother, faster. |
| **Adam** | per-parameter adaptive LR from running estimates of grad mean + variance. Robust default. |
| **AdamW** | Adam with *decoupled* weight decay. The standard for transformers. |

The from-scratch SGD step is one line, which is all you need to show you understand it:

```python
# conceptually, with torch.no_grad():
#   for p in model.parameters():
#       p -= lr * p.grad
```

In real code it is always:

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

opt.zero_grad()      # clear last step's gradients (they accumulate otherwise)
loss.backward()      # autograd fills p.grad for every parameter
opt.step()           # apply the update rule to every parameter
```

**The one gotcha to say out loud:** you must `zero_grad()` each step, because
`.backward()` *adds to* `.grad` rather than replacing it.

---

## 4. Schedulers — changing the learning rate over time (explain, do not drill)

The learning rate usually should not stay constant. A scheduler adjusts it across training.
Two patterns dominate modern training, and transformers essentially always use both
together:

- **Warmup** — start the LR near zero and ramp up over the first few hundred/thousand
  steps. Early gradients are noisy; warmup stops them from blowing up the fresh weights.
- **Cosine decay** — after warmup, smoothly anneal the LR down following a cosine curve to
  (near) zero by the end. Gives large steps early, fine steps late.

```python
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

for step in range(1000):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    sched.step()        # advance the LR schedule once per step
```

If asked "how would you train this," the strong answer is: **AdamW + linear warmup + cosine
decay**, which is the de facto recipe for transformers.

---

## 5. Putting it together — the training step

Every training loop, regardless of architecture, is this shape. If you can write it from
memory, you can train anything in the other notes:

```python
model = MyModel(...)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

for x, y in dataloader:
    opt.zero_grad()               # 1. clear old grads
    logits = model(x)             # 2. forward
    loss   = F.cross_entropy(logits, y)   # 3. loss (activation-free: CE wants logits)
    loss.backward()               # 4. autograd → grads
    opt.step()                    # 5. update weights
    sched.step()                  # 6. advance LR
```

---

## 6. Smoke test

```python
import torch
import torch.nn.functional as F

# activations return the right shapes / ranges
x = torch.randn(4, 8)
assert F.relu(x).shape == (4, 8)
assert (torch.sigmoid(x) > 0).all() and (torch.sigmoid(x) < 1).all()

# cross-entropy on logits
logits  = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))
assert F.cross_entropy(logits, targets).ndim == 0        # scalar

# InfoNCE: a perfect match (z2 == z1) should give near-zero loss
def info_nce(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    return F.cross_entropy(logits, torch.arange(z1.shape[0]))

z = torch.randn(16, 32)
matched   = info_nce(z, z.clone())          # positives are the diagonal → low loss
random    = info_nce(z, torch.randn(16, 32))
assert matched < random                     # matched pairs score better
print(f"matched loss {matched:.3f} < random loss {random:.3f}")

# one training step actually decreases a toy loss
w = torch.nn.Parameter(torch.randn(8, 1))
opt = torch.optim.AdamW([w], lr=0.1)
x = torch.randn(32, 8); y = torch.randn(32, 1)
l0 = F.mse_loss(x @ w, y)
for _ in range(50):
    opt.zero_grad(); F.mse_loss(x @ w, y).backward(); opt.step()
l1 = F.mse_loss(x @ w, y)
assert l1 < l0
print(f"loss {l0:.3f} -> {l1:.3f}")
print("SMOKE TEST PASSED")
```

---

## 7. The four in one screen

- **Activation:** nonlinearity between linears (else depth collapses). GELU/ReLU in
  hidden layers; softmax for categorical output / attention.
- **Loss:** the scalar you minimize. Cross-entropy wants *logits*. InfoNCE = cross-entropy
  over similarities, label = the diagonal.
- **Optimizer:** turns `.grad` into weight updates. AdamW is the transformer default;
  always `zero_grad()` first.
- **Scheduler:** moves the LR over time. Warmup + cosine decay is the standard recipe.

---

[← Back to contents](index.html) · [The PyTorch Module Pattern →](pytorch-module-pattern.html)
