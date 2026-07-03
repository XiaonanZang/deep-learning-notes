---
title: The PyTorch Module Pattern (Python classes for deep learning)
---

# The PyTorch Module Pattern

A focused tour of the one Python pattern every from-scratch PyTorch model uses:
subclass `nn.Module`, define your pieces in `__init__`, define the computation in
`forward`. This note explains `__init__`, `self`, `nn.Parameter`, `forward`, and how
`.parameters()` feeds the optimizer — using the exact classes from the
[Transformer](transformers.html) and [ViT](vision-transformers.html) notes as the
worked examples, so it is concrete, not abstract.

---

## 1. What problem the class solves

A neural network layer is two things bundled together:

- **State** — the learnable weights (a `Linear`'s matrix, an attention head's
  projections). This state must *persist* between calls and be *updated* by the optimizer.
- **Behavior** — the math that turns an input into an output using that state.

A plain function can express the behavior, but it has nowhere to keep the state. You
would have to pass every weight in by hand on every call and track them all yourself. A
**class** bundles the state (stored on the instance) with the behavior (its methods).
`nn.Module` is PyTorch's base class that adds three superpowers on top of a normal class:

1. It **finds all your weights automatically** so one call, `model.parameters()`, hands
   them to the optimizer.
2. It **moves them together** — `model.to("cuda")` moves every weight at once.
3. It **nests** — a module can hold other modules, and all of it registers automatically.

Everything below is how those three things work.

---

## 2. Python class mechanics, the minimum

Before the PyTorch part, the three words you asked about — `class`, `__init__`, `self` —
in the smallest possible example, no PyTorch.

```python
class Counter:
    # __init__ is the CONSTRUCTOR. It runs once, automatically, when you write
    # Counter(...). Its job is to set up the object's starting state.
    def __init__(self, start=0):
        # `self` is THE INSTANCE being built — the specific object. Assigning to
        # self.count creates an attribute that lives ON that object and persists
        # for its whole life. Every method receives `self` as its first argument,
        # which is how it reaches that stored state.
        self.count = start

    # A method. `self` is passed in automatically when you call c.tick().
    def tick(self):
        self.count += 1          # read + update state stored on the instance
        return self.count

c = Counter(start=10)   # __init__ runs, self.count = 10
c.tick()                # → 11   (state persisted between calls)
c.tick()                # → 12
```

Three ideas to carry forward:

| Word | What it is | In one line |
|---|---|---|
| `class` | a blueprint for objects | defines what state + methods every instance has |
| `__init__` | the constructor | runs at creation; sets up starting state via `self.` |
| `self` | the instance handle | how every method reaches this object's own state |

`Counter` already shows the core value: `self.count` **persists** across calls. A network
layer needs exactly that, except its persistent state is learnable weights.

---

## 3. The `nn.Module` contract

Every PyTorch layer or model follows the same four-part recipe:

```python
import torch
import torch.nn as nn

class MyLayer(nn.Module):              # 1. subclass nn.Module
    def __init__(self, D):
        super().__init__()             # 2. ALWAYS call this first
        self.linear = nn.Linear(D, D)  # 3. define pieces (they hold weights) in __init__

    def forward(self, x):              # 4. define the computation in forward
        return self.linear(x)
```

**1. Subclass `nn.Module`.** This is what grants the three superpowers from §1.

**2. `super().__init__()` — and why it must come first.** `nn.Module.__init__` sets up the
internal bookkeeping (empty registries for parameters and sub-modules) that make
auto-discovery work. If you assign `self.linear = ...` *before* calling it, PyTorch has
nowhere to register that layer and it silently vanishes from `.parameters()`. Rule: call
`super().__init__()` on the first line of every `__init__`, always.

**3. Define pieces in `__init__`.** Anything you assign to `self.` that is itself an
`nn.Module` (like `nn.Linear`) or an `nn.Parameter` gets **auto-registered**. You do not
maintain a list of weights; assigning them to `self` is the registration.

**4. `forward` — and why you never call it directly.** You write `model(x)`, not
`model.forward(x)`. `nn.Module` defines `__call__` (the method Python runs when you "call"
an object like a function), and `__call__` runs `forward` for you *plus* the hooks
autograd and other machinery rely on. Calling `.forward()` directly skips that machinery.
So: **define `forward`, but invoke with `model(x)`.**

```python
layer = MyLayer(D=64)
x = torch.randn(2, 10, 64)   # (B, T, D)
y = layer(x)                 # calls __call__ → forward. NOT layer.forward(x).
print(y.shape)               # torch.Size([2, 10, 64])
```

---

## 4. `nn.Parameter` vs. a plain tensor

A `nn.Parameter` is a tensor with one extra meaning: **"I am a learnable weight — track my
gradients and include me in `.parameters()`."** This is the difference between a value the
optimizer *updates* and a value that is just scratch data.

You saw two of these in the ViT note. They are raw parameters because there is no `nn.`
layer that produces them:

```python
# From the ViT note — the CLS token and the positional table.
self.cls = nn.Parameter(torch.zeros(1, 1, D))       # a learned vector
self.pos = nn.Parameter(torch.zeros(1, n_tokens, D)) # a learned table
```

Contrast the three ways a tensor can live on a module:

| Assignment | Registered as a weight? | Trained? | Moves with `.to()`? |
|---|---|---|---|
| `self.w = nn.Parameter(torch.randn(D))` | yes | yes | yes |
| `self.w = nn.Linear(D, D)` (wraps parameters) | yes (via the sub-module) | yes | yes |
| `self.buf = torch.randn(D)` (plain tensor) | **no** | no | **no** (a common bug) |

That last row is a classic trap: a plain tensor attribute is *not* registered, so it is
missing from `.parameters()` and is left behind on the CPU when you call `.to("cuda")`. If
you need a non-learned tensor that should still move with the model (like a fixed
sinusoidal table), use `self.register_buffer("pe", pe)` instead of a bare attribute.

Under the hood `nn.Linear` is just this pattern wrapping a weight and bias parameter:

```python
class Linear(nn.Module):                 # a from-scratch nn.Linear
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # nn.Parameter → these show up in .parameters() and get gradients.
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: (..., in_dim) → (..., out_dim). @ is matrix multiply; .T transposes weight.
        return x @ self.weight.T + self.bias
```

---

## 5. Modules nest — automatically

The real payoff: a module can hold other modules, and the whole tree registers itself.
This is exactly the `EncoderLayer` from the transformer note — it holds an attention
module and an FFN module, each of which holds `nn.Linear`s, each of which holds
parameters:

```python
class EncoderLayer(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.attn = MultiHeadAttention(D, h)   # a sub-module (holds 4 Linears)
        self.ffn  = FFN(D)                      # a sub-module (holds 2 Linears)
        self.ln1  = nn.LayerNorm(D)
        self.ln2  = nn.LayerNorm(D)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)   # sub-modules called like functions
        x = x + self.ffn(self.ln2(x))
        return x
```

Assigning `self.attn = MultiHeadAttention(...)` registers *the whole sub-tree*. One call
to `encoder_layer.parameters()` returns every weight underneath — the attention
projections, the FFN linears, the LayerNorm scales — with no manual list.

**When you have a variable number of sub-modules, use `nn.ModuleList`** (a plain Python
list would *not* register its contents):

```python
# From the transformer note's stacked encoder.
self.layers = nn.ModuleList([EncoderLayer(D, h) for _ in range(num_layers)])
# self.layers = [EncoderLayer(...) for ...]   # ← WRONG: a plain list hides the params
```

---

## 6. The lifecycle these superpowers unlock

Because every weight is registered, four everyday operations become one-liners:

```python
model = TransformerEncoder(vocab_size=10000, D=64, h=8, num_layers=4)

# 1. Hand every weight to the optimizer — this is why registration matters.
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Move the entire model (all sub-modules, all params) to a device at once.
model.to("cuda")

# 3. Switch behavior of dropout / batchnorm between training and inference.
model.train()   # training mode
model.eval()    # inference mode

# 4. Save / load all weights as one dictionary.
torch.save(model.state_dict(), "ckpt.pt")
model.load_state_dict(torch.load("ckpt.pt"))
```

The training step then reads the same every time:

```python
opt.zero_grad()          # clear old gradients
logits = model(x)        # forward  (calls __call__ → forward)
loss = loss_fn(logits, y)
loss.backward()          # autograd fills .grad on every registered Parameter
opt.step()               # optimizer updates every parameter using its .grad
```

`loss.backward()` and `opt.step()` only reach the weights *because* they were registered
by the `nn.Module` / `nn.Parameter` machinery. That is the whole point of the pattern.

---

## 7. End-to-end smoke test

A tiny two-layer module, from scratch, that you can run to see registration, forward, and
backward all working:

```python
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, D_in, D_hidden, D_out):
        super().__init__()                       # bookkeeping first
        self.fc1 = nn.Linear(D_in, D_hidden)     # sub-module → registered
        self.act = nn.GELU()
        self.fc2 = nn.Linear(D_hidden, D_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))   # invoked via __call__

model = TinyMLP(D_in=16, D_hidden=64, D_out=4)

# Registration check: parameters() finds every weight under the tree.
n_params = sum(p.numel() for p in model.parameters())
print("param tensors:", len(list(model.parameters())))   # 4  (2 weights + 2 biases)
print("total params:", n_params)                          # 16*64+64 + 64*4+4 = 1348

# Forward (note: model(x), not model.forward(x)).
x = torch.randn(8, 16)          # (B, D_in)
y = model(x)
print("output shape:", y.shape) # torch.Size([8, 4])
assert y.shape == (8, 4)

# Backward: gradients flow to every registered parameter.
loss = y.pow(2).mean()
loss.backward()
assert model.fc1.weight.grad is not None   # grad landed on a registered weight
print("SMOKE TEST PASSED")
```

---

## 8. The pattern in one screen

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()            # 1. always first — enables registration
        self.layer = nn.Linear(...)   # 2. sub-modules / nn.Parameter in __init__
        self.weight = nn.Parameter(...)

    def forward(self, x):             # 3. the computation
        return self.layer(x)

model = MyModel(...)                  # __init__ runs
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # 4. params → optimizer
y = model(x)                          # 5. call the object, not .forward()
loss.backward(); opt.step()           # 6. autograd + update reach registered weights
```

If you can write that from memory, every from-scratch answer in the other notes has a home
to live in. The architecture is the interesting part; this is the frame that holds it.

---

[← Back to contents](index.html) · [Transformer Architecture →](transformers.html)
