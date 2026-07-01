---
title: Transformer Architecture
---

# Transformer Architecture

A plain-language tour of the transformer encoder, followed by a from-scratch,
fully commented PyTorch implementation that runs end to end — from integer token
IDs to contextual output vectors.

---

## 1. What problem it solves

Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time.
Each step depended on the previous hidden state, which meant: no parallelism, and
long-range dependencies decayed or vanished. The transformer replaced recurrence
with **attention** — every token can directly attend to every other token in one
matrix operation, regardless of distance.

### The classical example: text query

Sentence: **"The animal didn't cross the street because it was too tired."**

What does "it" refer to? A human resolves this instantly by attending to "animal"
and "street" and reading the semantics. A transformer does the same thing
mechanically: the token "it" issues a query, every other token responds with a key,
and the dot product scores tell the model that "animal" is the most relevant referent.

---

## 2. Architecture overview

A transformer encoder is a stack of identical layers. Each layer has two sub-layers:

```
Input X
  │
  ├── [Sub-layer 1] Multi-Head Self-Attention
  │     └── Residual + LayerNorm  →  X + Attention(X)
  │
  └── [Sub-layer 2] Feed-Forward Network (FFN)
        └── Residual + LayerNorm  →  X + FFN(X)
```

**Multi-Head Self-Attention** — runs h attention heads in parallel, each attending
with its own Q/K/V projections. Heads learn different relationship types (syntax,
coreference, proximity). Outputs are concatenated and projected back to D.

**FFN** — two linear layers with a nonlinearity (ReLU or GELU) in between. Applied
independently per token. Expands to 4D internally, contracts back to D:

```
FFN(x) = Linear(GELU(Linear(x, D → 4D)), 4D → D)
```

**Residual connections** — add the input back to the sub-layer output. Prevents
vanishing gradients and lets lower layers preserve raw token information.

**LayerNorm** — normalizes across the D dimension per token (not across the batch).
Stabilizes training.

### Full encoder block: shape flow

```
X:          (B, T, D)
  → MHA:    (B, T, D)   [h heads, each D/h, concat + W_O]
  → + X:    (B, T, D)   residual
  → LN:     (B, T, D)
  → FFN:    (B, T, D)   [expand to 4D, contract back]
  → + prev: (B, T, D)   residual
  → LN:     (B, T, D)   output of one encoder layer
```

### Positional encoding

Attention itself is permutation-invariant — it sees a set, not a sequence.
Positional encodings inject order. The original paper used fixed sinusoids (one
frequency per dimension); modern models learn positional embeddings or use rotary
(RoPE) / ALiBi variants.

---

## 3. PyTorch implementation — step by step

```python
import torch
import torch.nn as nn
import math
```

---

### Step 1: input → token embeddings

The raw input is a sequence of integer token IDs — one integer per word (or subword).
The embedding layer is a lookup table: each integer maps to a learned vector of size
D. This is the first place the model turns discrete symbols into continuous vectors
it can compute with.

```python
# Input token IDs: (B, T)
#   B = batch size (number of sentences processed in parallel)
#   T = sequence length (number of tokens per sentence)
#   Values are integers in [0, vocab_size)

token_ids = torch.tensor([[4, 17, 2, 9, 0],
                           [6,  3, 8, 1, 5]])   # (B=2, T=5)

# Embedding lookup: maps each integer to a learnable D-dimensional vector.
# After lookup, each token is represented as a dense vector — the model can now do math on it.
embedding = nn.Embedding(num_embeddings=10000,  # vocab_size: total number of unique tokens
                         embedding_dim=64)       # D: size of each token vector

token_emb = embedding(token_ids)   # (B, T, D) = (2, 5, 64)
```

---

### Step 2: positional encoding → add to embeddings

Self-attention is permutation-invariant: it sees a bag of vectors, not a sequence.
Without positional information, "cat sat mat" and "mat sat cat" produce identical
attention patterns. Positional encoding injects the position of each token into its
embedding before anything enters the encoder.

The original paper uses fixed sinusoidal encodings — no learned parameters. Even
positions use `sin`, odd positions use `cos`, each at a different frequency. The
resulting vector is unique for every position and smooth across neighboring positions.

```python
def sinusoidal_encoding(T, D):
    # Build a (T, D) matrix where each row is the positional encoding for one position.
    pe = torch.zeros(T, D)

    pos = torch.arange(T).unsqueeze(1)          # (T, 1) — position indices 0, 1, ..., T-1

    # Frequency term: one frequency per pair of dimensions.
    # dim_i ranges over even indices 0, 2, 4, ..., D-2.
    # The formula 1 / 10000^(2i/D) gives a geometric progression of frequencies.
    dim_i = torch.arange(0, D, step=2)          # (D/2,)
    freq  = 1.0 / (10000 ** (dim_i / D))        # (D/2,) — high freq for small i, low freq for large i

    pe[:, 0::2] = torch.sin(pos * freq)         # even dimensions  → sin
    pe[:, 1::2] = torch.cos(pos * freq)         # odd dimensions   → cos

    return pe                                    # (T, D)

T, D = token_emb.shape[1], token_emb.shape[2]

pos_enc = sinusoidal_encoding(T, D)             # (T, D)

# Add positional encoding to token embeddings.
# pos_enc is (T, D); token_emb is (B, T, D). Broadcasting adds pos_enc to every batch item.
x = token_emb + pos_enc                         # (B, T, D) — now carries both meaning and position
```

---

### Step 3: stable softmax

| | |
|---|---|
| **What it is** | A function that converts a vector of arbitrary real numbers into a probability distribution — all values positive, summing to 1. |
| **Where it is used** | Inside scaled dot-product attention, applied row-wise to the score matrix `S` of shape `(B, T, T)`. Each row represents one token's raw affinity scores toward all T positions; softmax turns that row into attention weights that sum to 1. |
| **Formula** | `softmax(x_i) = exp(x_i) / Σ exp(x_j)` — exponential of each element divided by the sum of all exponentials. |
| **Stable variant** | `softmax(x_i) = exp(x_i − max(x)) / Σ exp(x_j − max(x))` — subtracting the row max before exp. Mathematically identical (the constant cancels in numerator and denominator), but prevents `exp` from overflowing to `inf` when scores are large. |
| **Why this function** | Exponential amplifies differences: a token with a slightly higher score gets disproportionately more weight, producing sharp, selective attention. A linear normalization (divide by sum) would not amplify — attention would be diffuse and fail to focus. The softmax exponent is what makes the model able to pick a dominant token rather than averaging everything equally. |

```python
def softmax(x, dim=-1):
    # Subtract the max along the target dimension before applying exp.
    # This is the numerical stability fix: exp(x - max) never overflows
    # because the largest input to exp is always 0.
    # keepdim=True preserves the shape so broadcasting works correctly.
    x = x - x.max(dim=dim, keepdim=True).values

    e = torch.exp(x)           # exp of each element; now safe because inputs are <= 0

    return e / e.sum(dim=dim, keepdim=True)   # normalize so each row sums to 1
```

---

### Step 4: embeddings → attention matrix (scaled dot-product attention)

This is where `x` — the combined token + positional embedding — enters the attention
mechanism. Three linear projections produce Q, K, V. The dot product of Q and K^T
produces the raw score matrix: how much each token attends to every other token.
Scaling and softmax convert those raw scores into the final attention weight matrix.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (B, T, Dk)  — what each token is looking for
    # K: (B, T, Dk)  — what each token offers as a key
    # V: (B, T, Dv)  — what each token offers as content
    Dk = Q.shape[-1]

    # Raw score matrix: dot product of every query with every key.
    # Q @ K^T gives a (T, T) matrix per batch item:
    #   scores[b, i, j] = how much token i in batch b attends to token j.
    scores = Q @ K.transpose(-2, -1)      # (B, T, T)

    # Scale by sqrt(Dk) to prevent the dot products from growing large as Dk increases,
    # which would push softmax into regions of near-zero gradients.
    scores = scores / math.sqrt(Dk)       # (B, T, T)

    # Optional causal mask: set future positions to -inf before softmax.
    # softmax(-inf) → 0, so the model cannot attend to future tokens.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax row-wise: each row becomes a probability distribution over T positions.
    # This is the attention weight matrix A — the "how much to attend" answer.
    A = softmax(scores, dim=-1)           # (B, T, T)

    # Weighted sum of value vectors: each output token is a blend of all value vectors,
    # weighted by how much it attended to each position.
    out = A @ V                           # (B, T, Dv)

    return out, A   # return both output and attention weights (A useful for inspection)
```

---

### Step 5: multi-head attention

Running one attention head uses the full D dimensions for a single relationship type.
Multi-head attention splits D into h smaller subspaces and runs h independent attention
operations in parallel. Each head can specialize: one might track syntax, another
coreference, another proximity. Outputs are concatenated and projected back to D.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        assert D % h == 0, "D must be divisible by h so each head gets equal dimensions"
        self.h  = h
        self.Dk = D // h                  # each head operates in a Dk-dimensional subspace

        # One linear projection per role (Q, K, V), sized D → D.
        # Each projection covers all h heads at once (fused for efficiency).
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)   # output projection: concat of h heads → D

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Project x to Q, K, V — each (B, T, D).
        # Then reshape to split D into h heads of size Dk:
        #   (B, T, D) → (B, T, h, Dk) → (B, h, T, Dk)
        # Transposing puts h second so each head is a contiguous (T, Dk) slice.
        Q = self.W_Q(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)
        K = self.W_K(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)
        V = self.W_V(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)

        # Run scaled dot-product attention for all h heads simultaneously.
        # The function treats the h dimension as a batch dimension — no explicit loop needed.
        out, _ = scaled_dot_product_attention(Q, K, V, mask)           # (B, h, T, Dk)

        # Concatenate heads: reverse the transpose, then flatten h and Dk back into D.
        # .contiguous() is required because transpose changes strides but not memory layout;
        # .view() requires contiguous memory to reshape safely.
        out = out.transpose(1, 2).contiguous().view(B, T, D)           # (B, T, D)

        # Final linear projection mixes information across heads.
        return self.W_O(out)                                            # (B, T, D)
```

---

### Step 6: feed-forward network (FFN)

After attention, each token has gathered information from the rest of the sequence.
The FFN processes each token independently to transform that gathered information into
a richer representation. It expands to 4D internally (more capacity), applies a
nonlinearity, then contracts back to D.

```python
class FFN(nn.Module):
    def __init__(self, D, expansion=4):
        super().__init__()
        self.fc1  = nn.Linear(D, D * expansion)   # expand: D → 4D
        self.act  = nn.GELU()                      # GELU: smoother than ReLU, standard in modern transformers
        self.fc2  = nn.Linear(D * expansion, D)   # contract: 4D → D

    def forward(self, x):
        # x: (B, T, D) — applied identically and independently to each token position
        x = self.fc1(x)    # (B, T, 4D)
        x = self.act(x)    # (B, T, 4D) — nonlinearity; without this, two linears collapse to one
        x = self.fc2(x)    # (B, T, D)
        return x
```

---

### Step 7: full encoder layer (residuals + LayerNorm)

One encoder layer wraps attention and FFN with two structural patterns that make deep
stacking stable:

- **Residual connection**: adds the input back to the sub-layer output (`x + sublayer(x)`).
  Gradients flow directly through the addition, bypassing the sub-layer — prevents
  vanishing gradients in deep stacks.
- **LayerNorm**: normalizes across the D dimension per token. Keeps activations in a
  stable range regardless of depth.

```python
class EncoderLayer(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.attn = MultiHeadAttention(D, h)
        self.ffn  = FFN(D)
        self.ln1  = nn.LayerNorm(D)    # applied before attention (pre-norm order)
        self.ln2  = nn.LayerNorm(D)    # applied before FFN

    def forward(self, x, mask=None):
        # Pre-norm order: normalize x before passing into the sub-layer.
        # Original paper used post-norm (normalize after); pre-norm trains more stably in practice.

        # Sub-layer 1: multi-head attention with residual
        x = x + self.attn(self.ln1(x), mask)   # (B, T, D)

        # Sub-layer 2: FFN with residual
        x = x + self.ffn(self.ln2(x))          # (B, T, D)

        return x
```

---

### Step 8: stacked encoder + end-to-end smoke test

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, D, h, num_layers, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D)              # token ID → dense vector
        self.pos_enc   = sinusoidal_encoding(max_len, D)          # fixed; not a learned parameter
        self.layers    = nn.ModuleList([EncoderLayer(D, h) for _ in range(num_layers)])

    def forward(self, token_ids, mask=None):
        # token_ids: (B, T) — integer token indices
        T = token_ids.shape[1]

        x = self.embedding(token_ids)                             # (B, T, D) token embeddings
        x = x + self.pos_enc[:T].to(x.device)                    # (B, T, D) add positional encoding

        for layer in self.layers:                                 # pass through each encoder layer
            x = layer(x, mask)

        return x                                                  # (B, T, D) contextual representations


# --- Smoke test ---
B, T       = 2, 10          # batch of 2 sentences, each 10 tokens long
vocab_size = 10000
D, h       = 64, 8          # model dim 64, 8 attention heads (Dk = 8 per head)
num_layers = 4

token_ids = torch.randint(0, vocab_size, (B, T))   # (B, T) random token indices

model = TransformerEncoder(vocab_size=vocab_size, D=D, h=h, num_layers=num_layers)
out   = model(token_ids)

print(out.shape)            # expect: torch.Size([2, 10, 64])
assert out.shape == (B, T, D)
```

---

[← Back to contents](index.html) · [Vision Transformers →](vision-transformers.html)
