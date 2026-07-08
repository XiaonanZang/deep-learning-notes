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

### Input representation: the (B, T, D) tensor

Everything above operates on `X`, a tensor of shape `(B, T, D)`. Before any layer
runs, raw text is turned into that tensor. The three axes are built by three different
steps:

| Axis | Name | Meaning | Set by |
|---|---|---|---|
| B | batch | sequences processed in parallel (think: sentences) | how many inputs you group |
| T | sequence length | number of tokens in each sequence | the tokenizer |
| D | model dimension | length of each token's feature vector | you (a hyperparameter) |

Two distinct mechanisms build the last two axes.

**The tokenizer builds T.** It splits raw text into tokens, which are subword units,
not whole words. One word can become several tokens:

```
"cat"          → ["cat"]              (1 token)
"didn't"       → ["did", "n't"]       (2 tokens)
"tokenization" → ["token", "ization"] (2 tokens)
```

Each token maps to an integer ID from the vocabulary, so a sentence becomes a list of
integers of shape `(B, T)`. T is therefore input-dependent: a longer sentence has a
larger T.

**The embedding builds D.** Each token ID indexes one row of a learned lookup table of
shape `(vocab_size, D)`; that row is the token's D-dimensional vector. D is not read
from the text. You choose it (64, 512, 768...), and the table values are learned during
training.

```
"the cat sat"  --tokenizer-->  [4, 17, 2]   --embedding-->  (T=3, D=64)
   raw text                    ids (B, T)                   vectors (B, T, D)
```

So the tokenizer makes the T axis (how many tokens), the embedding makes the D axis
(how wide each token is). From here on every sub-layer takes `(B, T, D)` in and returns
`(B, T, D)` out; only the attention weight matrix is `(B, T, T)`.

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

### Batching variable-length sequences

Because T is set by the tokenizer, two sentences rarely share the same length, yet a
tensor must be rectangular: every row in a batch has to share one T. Two steps
reconcile this.

**1. Pad to a common length.** Take the longest sequence in the batch and fill the
shorter ones with a special `[PAD]` token (usually id `0`) until all match:

```
"the cat sat"          → [4, 17, 2]      → [4, 17, 2, 0, 0]
"the animal was tired" → [4, 31, 8, 25]  → [4, 31, 8, 25, 0]
                                            (B=2, T=5)
```

**2. Mask the padded positions.** Padding is filler, so real tokens must not attend to
it. Before softmax, the scores at pad positions are set to `-inf`; softmax turns those
into 0 weight. This is the same `masked_fill` mechanism used for causal masking:

```python
scores = scores.masked_fill(mask == 0, float('-inf'))   # -inf → softmax → 0
```

The one mechanism supports two different masks:

| Mask | Hides | Purpose |
|---|---|---|
| Padding mask | `[PAD]` filler positions | batch sentences of different lengths together |
| Causal mask | future positions | stop a decoder attending ahead when generating |

Real models often apply both at once.

**Practical variants:**
- **Truncation:** if a sequence exceeds a chosen `max_len`, cut it, so T per batch is
  `min(longest sequence, max_len)`.
- **Length bucketing:** group similar-length sequences into the same batch to minimize
  padding and wasted compute.

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
    #
    # Two separate jobs here:
    #   dim=-1      → chooses WHICH axis to reduce. Reducing the last axis gives
    #                 one max per row (the max across columns). This is what makes
    #                 the operation row-wise.
    #   keepdim=True → controls only the SHAPE of the result, not its values.
    #                 It keeps the reduced axis as size 1 instead of dropping it:
    #                   x is           (B, T, T)
    #                   max, keepdim   (B, T, 1)   ← broadcasts back over the columns
    #                   max, no keepdim(B, T)      ← would misalign and break the subtract
    #                 The trailing 1 lets every element in a row subtract THAT row's max.
    x = x - x.max(dim=dim, keepdim=True).values

    e = torch.exp(x)           # exp of each element; now safe because inputs are <= 0

    # Same dim + keepdim pattern for the normalizer: sum across the row (dim=-1),
    # keep shape (B, T, 1) so each element divides by its own row's sum.
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
    #
    # transpose(-2, -1) swaps ONLY the last two axes (T and Dk), turning
    # K: (B, T, Dk) into K^T: (B, Dk, T) so the matmul lines up:
    #   Q (B, T, Dk) @ K^T (B, Dk, T) → (B, T, T).
    # The batch axis B is deliberately left untouched. Negative indices are used
    # (not .T, which would reverse ALL axes and move B) so the same line also works
    # when Q/K are 4-D (B, h, T, Dk) in multi-head: it still swaps just the last two.
    scores = Q @ K.transpose(-2, -1)      # (B, T, T)

    # Scale by sqrt(Dk). This is a DIFFERENT fix from softmax's max-subtraction:
    #   - max-subtraction only guards exp() against overflow; it is mathematically
    #     identical to plain softmax, so it does NOT change the attention distribution.
    #   - sqrt(Dk) scaling DOES change the distribution: it controls how peaked softmax is.
    # Why sqrt(Dk): a score is a dot product of two Dk-dim vectors. With ~unit-variance,
    # independent components, that sum of Dk terms has variance ~Dk, so std ~ sqrt(Dk).
    # Scores therefore grow like sqrt(Dk); large scores saturate softmax into a near
    # one-hot spike where its gradient is ~0 (vanishing gradients, learning stalls).
    # Dividing by sqrt(Dk) renormalizes the score variance back to ~1 regardless of Dk,
    # keeping softmax in a smooth, well-conditioned region. So both are needed:
    # max-subtraction keeps the math from overflowing, sqrt(Dk) keeps the gradient alive.
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
        # // is integer division (D / h would give a float, e.g. 8.0). Dk is a dimension
        # size, so it must be an int: it feeds .view(B, T, h, Dk) below, and tensor shapes
        # reject floats. The assert above guarantees the division is exact, so // loses
        # nothing here; it just keeps the type correct for the reshape.
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
        #
        # How view() splits: it does not "choose" an axis. It reinterprets the same flat,
        # row-major buffer into the new shape, total element count preserved (D = h*Dk).
        # Row-major means the LAST axis is filled first, then the second-to-last, etc.
        # (odometer order: rightmost index moves fastest, stride 1). Here only the trailing
        # D is refactored into (h, Dk), so each token's D features split into h consecutive
        # Dk-chunks: feature d → head = d // Dk, position-in-head = d % Dk.
        # Factor ORDER matters: (h, Dk) makes each head a contiguous Dk slice; (Dk, h) would
        # interleave across heads instead.
        # Transposing puts h second so each head is a contiguous (T, Dk) slice.
        Q = self.W_Q(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)
        K = self.W_K(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)
        V = self.W_V(x).view(B, T, self.h, self.Dk).transpose(1, 2)   # (B, h, T, Dk)

        # Run scaled dot-product attention for all h heads simultaneously.
        # The function treats the h dimension as a batch dimension — no explicit loop needed.
        out, _ = scaled_dot_product_attention(Q, K, V, mask)           # (B, h, T, Dk)

        # Concatenate heads: reverse the transpose, then flatten h and Dk back into D.
        # Why .contiguous() is needed here: view() only works on a buffer whose memory is
        # already laid out in row-major order for the current shape. transpose() does NOT
        # move data; it just swaps the shape/stride metadata, so after transpose(1, 2) the
        # tensor is non-contiguous (strides no longer match row-major). Calling view()
        # directly would raise a runtime error. .contiguous() physically re-copies the data
        # into row-major layout so the following view(B, T, D) can pour it back correctly.
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

**Division of labor:** attention is *communication* (tokens mix across the T axis); the
FFN is *computation* (each token is processed on its own, no mixing). Most of the model's
learned capacity lives in these two FFN matrices — in large models roughly two-thirds of
all parameters sit here.

```python
class FFN(nn.Module):
    def __init__(self, D, expansion=4):
        super().__init__()
        # Expand to 4D: a wider hidden space gives the nonlinearity room to represent
        # richer per-token functions. 4x is an empirical convention (the original
        # Transformer used 512 → 2048), a good capacity/compute tradeoff, not a derived
        # constant. This is where much of the model's parameters/"knowledge" live.
        self.fc1  = nn.Linear(D, D * expansion)   # expand: D → 4D
        self.act  = nn.GELU()                      # GELU: smoother than ReLU, standard in modern transformers
        # Contract back to D so the output shape chains into the residual and next layer.
        # Note the minimal unit is linear → nonlinearity → linear: WITHOUT the GELU the two
        # linears collapse into a single linear map (no added power). Real depth comes from
        # STACKING encoder layers (Step 8), not from deepening this FFN, so each block stays
        # shallow and uniform, then repeats N times.
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
        # register_buffer, NOT a plain attribute. pos_enc is a fixed (non-learned) tensor, but it must
        # still live on the same device as the model. A plain `self.pos_enc = tensor` is neither a
        # Parameter nor a buffer, so model.to('cuda') would NOT move it — it would stay on CPU and cause
        # a device-mismatch RuntimeError at the `x + pos_enc` step below. Registering it as a buffer means
        # .to(device), .half(), and save/load all carry it along automatically — the same mechanism that
        # moves BatchNorm's running_mean/running_var. Rule: fixed tensors a module needs → register_buffer.
        self.register_buffer('pos_enc', sinusoidal_encoding(max_len, D))   # fixed positional table (not learned)
        self.layers    = nn.ModuleList([EncoderLayer(D, h) for _ in range(num_layers)])

    def forward(self, token_ids, mask=None):
        # token_ids: (B, T) — integer token indices
        T = token_ids.shape[1]

        x = self.embedding(token_ids)                             # (B, T, D) token embeddings
        # No .to(x.device) needed: pos_enc is a registered buffer, so it already sits on the model's
        # device after model.to(...). (Were it a plain tensor attribute, this line would crash on GPU.)
        x = x + self.pos_enc[:T]                                  # (B, T, D) add positional encoding

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
