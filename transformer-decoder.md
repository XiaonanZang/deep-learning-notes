---
title: The Transformer Decoder
---

# The Transformer Decoder

The [encoder note](https://xiaonanzang.github.io/deep-learning-notes/transformers.html)
built a stack that turns a sequence into contextual vectors, every token free to look at
every other token. The **decoder** is the half that *generates* — it produces a sequence one
token at a time, and to do that it changes attention in exactly two ways. This note builds
those two changes from scratch, reuses the encoder's multi-head attention verbatim, and ends
with a runnable autoregressive generation loop and the one optimization (KV cache) that makes
generation practical.

Everything here runs. The naming matches the encoder note (`D`, `h`, `Dk`,
`MultiHeadAttention`) so the two pages compose.

---

## 1. What the decoder adds over the encoder

Two structural changes, and nothing else about the attention primitive changes:

| Piece | Encoder | Decoder |
|---|---|---|
| Self-attention | bidirectional — every token sees every token | **causal / masked** — a token sees only itself and the past |
| Cross-attention | — | **new** — queries come from the decoder, keys/values from the *encoder output* |
| Sub-layers per block | 2 (self-attn, FFN) | **3** (masked self-attn, cross-attn, FFN) |

The reason for the mask is the whole point of a generator: at training time the decoder is
shown the full target sequence in parallel, but it must never use a future token to predict
the present one, because at inference time the future does not exist yet. The mask enforces
that the parallel training path matches the one-token-at-a-time inference path.

Cross-attention is where the decoder *reads the input*: in translation, the encoder digests
the source sentence into `enc_out`, and every decoder position queries that memory to decide
what to emit next.

---

## 2. The causal (look-ahead) mask

A causal mask is a lower-triangular matrix of ones. Position *i* (a query) is allowed to
attend to key *j* only when `j <= i`. Everything above the diagonal — the future — is set to
`0`, and the attention step fills those with `-inf` **before** softmax so they receive exactly
zero weight.

```python
import torch

def causal_mask(T):
    # (T, T): 1 = keep (j <= i, the past and present), 0 = block (j > i, the future).
    # torch.tril keeps the lower triangle (including the diagonal) and zeros the rest.
    return torch.tril(torch.ones(T, T))

# causal_mask(4):
# [[1, 0, 0, 0],   position 0 sees only position 0
#  [1, 1, 0, 0],   position 1 sees 0,1
#  [1, 1, 1, 0],   position 2 sees 0,1,2
#  [1, 1, 1, 1]]   position 3 sees everything so far
```

The masking itself lives inside scaled dot-product attention — identical to the encoder's,
with one `masked_fill` line. **Placement matters:** the mask goes on the raw scores, *before*
softmax. Fill the blocked positions with `-inf`; then `exp(-inf) = 0`, so those keys drop out
and every row still sums to 1. Masking *after* softmax would corrupt an already-normalized
distribution.

```python
import math

def stable_softmax(x, dim=-1):
    x_max = x.max(dim, keepdim=True).values      # subtract the row max for numerical stability
    e = torch.exp(x - x_max)
    return e / e.sum(dim, keepdim=True)          # normalize over the key axis

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (B, h, Tq, Dk)   K, V: (B, h, Tk, Dk)
    Dk = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Dk)     # (B, h, Tq, Tk) similarity of each query to each key
    if mask is not None:
        # BEFORE softmax, on the scores. -inf -> 0 weight after exp; the mask broadcasts over (B, h).
        scores = scores.masked_fill(mask == 0, float('-inf'))
    A = stable_softmax(scores, dim=-1)                    # (B, h, Tq, Tk) attention weights, rows sum to 1
    return A @ V, A                                       # (B, h, Tq, Dk) values, plus the weights
```

---

## 3. One attention module, two wirings

Here is the single idea that makes the decoder cheap to build: **self-attention and
cross-attention are the same operation.** The only difference is where the queries come from
versus where the keys and values come from. So we generalize the encoder's `MultiHeadAttention`
to take a query source and an (optional) separate key/value source.

- **Self-attention:** `x_kv is None` → keys/values come from the same `x_q`. Used with a causal
  mask in the decoder.
- **Cross-attention:** `x_q` is the decoder state, `x_kv` is the encoder output. No causal mask —
  the decoder may read the entire input.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        assert D % h == 0, "D must be divisible by h"
        self.h, self.Dk = h, D // h
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)

    def _heads(self, t):                                  # (B, L, D) -> (B, h, L, Dk)
        B, L, D = t.shape
        return t.view(B, L, self.h, self.Dk).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:                                  # self-attention: keys/values from the queries' own sequence
            x_kv = x_q
        B, Tq, D = x_q.shape
        Q = self._heads(self.W_Q(x_q))                    # (B, h, Tq, Dk)  queries from x_q
        K = self._heads(self.W_K(x_kv))                   # (B, h, Tk, Dk)  keys   from x_kv
        V = self._heads(self.W_V(x_kv))                   # (B, h, Tk, Dk)  values from x_kv
        out, _ = scaled_dot_product_attention(Q, K, V, mask)   # (B, h, Tq, Dk)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)  # merge heads -> (B, Tq, D)
        return self.W_O(out)
```

When `Tq != Tk` (six decoder positions querying five encoder positions, say), the score matrix
`(B, h, Tq, Tk)` is simply **rectangular**. Nothing else changes. That rectangle is the entire
mechanism of cross-attention.

The FFN is unchanged from the encoder — expand to 4D, nonlinearity, contract back:

```python
class FFN(nn.Module):
    def __init__(self, D, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(D, expansion * D)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * D, D)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

---

## 4. The decoder layer — three sub-layers

An encoder-decoder decoder layer stacks three residual sub-layers, pre-norm (LayerNorm before
each sub-layer, same convention as the encoder note):

1. **Masked self-attention** — the target attends to itself, causally.
2. **Cross-attention** — the target queries the encoder memory.
3. **FFN** — per-token computation.

```python
class DecoderLayer(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.self_attn  = MultiHeadAttention(D, h)   # causal, over the target sequence
        self.cross_attn = MultiHeadAttention(D, h)   # Q = target, K/V = encoder output
        self.ffn = FFN(D)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, x, enc_out, look_ahead):
        # 1) masked self-attention: each target token gathers from earlier target tokens only
        x = x + self.self_attn(self.ln1(x), mask=look_ahead)
        # 2) cross-attention: each target token reads from the whole encoded input (no mask)
        x = x + self.cross_attn(self.ln2(x), x_kv=enc_out)
        # 3) position-wise FFN
        x = x + self.ffn(self.ln3(x))
        return x

# smoke test
B, T_dec, T_enc, D, h = 2, 6, 5, 32, 4
tgt, enc_out = torch.randn(B, T_dec, D), torch.randn(B, T_enc, D)
print(DecoderLayer(D, h)(tgt, enc_out, causal_mask(T_dec)).shape)   # torch.Size([2, 6, 32])
```

### What exactly is `enc_out`?

"The encoder output" is ambiguous, so pin it down: **`enc_out` is the entire `(B, T_src, D)` output
of the *top* (last) encoder layer — one contextual vector per source token.** It is *not* a pooled
summary, *not* the CLS token, *not* a middle layer. In cross-attention it is projected into the keys
and values (`W_K(enc_out)`, `W_V(enc_out)`), so there are `T_src` keys/values — one per source token —
and the target's queries attend over all of them.

Two facts that resolve most of the confusion:

- **It is computed once.** The source runs through the whole encoder stack; the final layer's output
  *is* `enc_out`. It does not change per decoder step or per decoder layer.
- **The same `enc_out` feeds every decoder layer's cross-attention.** One shared source-side memory,
  read by the entire decoder stack.

```python
# enc_out: the encoder's full final-layer sequence — one vector per source token.
enc_out = torch.randn(B, T_enc, D)          # (B, T_src, D)  (here a stand-in for the encoder output)

layers = [DecoderLayer(D, h) for _ in range(2)]
y = tgt
for layer in layers:
    y = layer(y, enc_out, causal_mask(T_dec))   # SAME enc_out passed into every decoder layer
    # Q length follows the target (T_dec); K/V length follows enc_out (T_enc) -> rectangular scores
print("target:", tuple(tgt.shape), " enc_out:", tuple(enc_out.shape), " out:", tuple(y.shape))
# target: (2, 6, 32)  enc_out: (2, 5, 32)  out: (2, 6, 32)
```

**Translation picture:** source *"je suis étudiant"* (3 tokens) → `enc_out` is **3 vectors** `(B, 3, D)`,
each a context-aware representation of one French token. As the decoder generates English, every
target position's cross-attention looks at those 3 French vectors to decide what to emit. `enc_out`
*is* those vectors — the full set, not a selection.

---

## 5. Decoder-only (GPT) vs encoder-decoder

Two families, and the difference is whether an encoder exists at all:

| | Encoder-decoder | Decoder-only (GPT) |
|---|---|---|
| Example | translation, T5, the original Transformer | GPT, Llama, most modern LLMs |
| Sub-layers per block | 3 (masked self-attn, cross-attn, FFN) | **2** (masked self-attn, FFN) |
| Cross-attention | yes — reads a separate encoded input | **none** — there is no separate input to read |
| Input handling | source goes through the encoder; target through the decoder | prompt and continuation are one sequence; the prompt is just the first tokens |

A decoder-only block is a decoder layer with the cross-attention sub-layer deleted. Everything
that makes it a *decoder* — the causal mask — stays.

```python
class GPTBlock(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.self_attn = MultiHeadAttention(D, h)    # causal self-attention only
        self.ffn = FFN(D)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(self, x, look_ahead):
        x = x + self.self_attn(self.ln1(x), mask=look_ahead)   # no cross-attention
        x = x + self.ffn(self.ln2(x))
        return x
```

This is why "everything is a decoder-only transformer" now: for time-series or language, a
single causal stack over one sequence, trained to predict the next token, is enough. There is
no separate input to encode — the context *is* the earlier part of the sequence.

---

## 6. Autoregressive generation

Generation is a loop: run the model on the tokens so far, take the logits at the **last**
position, pick the next token, append it, and feed the longer sequence back in. "Autoregressive"
means each new token is conditioned on all the tokens already produced.

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab, D, h, L, max_T=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(max_T, D)
        self.blocks = nn.ModuleList([GPTBlock(D, h) for _ in range(L)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab)            # project D -> vocab logits (the "LM head")
    def forward(self, ids):                        # ids: (B, T)
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T))
        m = causal_mask(T)
        for blk in self.blocks:
            x = blk(x, m)
        return self.head(self.norm(x))             # (B, T, vocab)

@torch.no_grad()
def generate(model, ids, n_new):
    for _ in range(n_new):
        logits = model(ids)                        # (B, T, vocab) — recomputes ALL T positions
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)   # greedy: highest-prob token at the last position
        ids = torch.cat([ids, nxt], dim=1)         # append and feed back in
    return ids

# smoke test
gpt = TinyGPT(vocab=50, D=32, h=4, L=2)
start = torch.randint(0, 50, (1, 3))
print(tuple(generate(gpt, start, n_new=5).shape))   # (1, 8): 3 prompt + 5 generated
```

(Greedy `argmax` here for determinism; real generation samples from the softmax with a
temperature, or top-k / top-p.)

---

## 7. KV cache — why it matters

Look again at `generate`: each step calls `model(ids)` on the **entire** growing sequence. Step
*t* recomputes the keys and values for all *t* positions, even though positions `0..t-1` have
not changed. That makes naive generation cost O(T²) in redundant work.

The fix is the **KV cache**. Because a token's key and value depend only on that token (and its
position), never on later tokens, they can be computed once and stored. On each step you:

1. compute Q, K, V for **only the one new token**,
2. append its K and V to the cached K and V from all previous steps,
3. attend: the single new query against the full cached keys/values.

Per step the work drops from "re-encode T tokens" to "encode 1 token, attend over T" — the
difference between quadratic and linear generation, and the reason serving stacks track cache
memory so carefully (the cache grows with sequence length × layers × heads). It changes *nothing*
about the math or the output; it only removes recomputation. The causal mask is also unnecessary
during cached generation — the new query is the last position, so it can legally see the entire
cache.

Made concrete: add an incremental `step` path that takes **one** new token plus the layer's
cached `(K, V)`, appends this token's key/value, and attends the single new query over the whole
cache. No mask — the one query is the newest position, so it may see everything stored.

```python
class MultiHeadAttention(MultiHeadAttention):        # add a step() method to the class above
    def step(self, x_new, cache):                    # x_new: (B, 1, D); cache: (K, V) or None
        B, _, D = x_new.shape
        q = self._heads(self.W_Q(x_new))             # (B, h, 1, Dk)  query for the new token only
        k = self._heads(self.W_K(x_new))             # (B, h, 1, Dk)
        v = self._heads(self.W_V(x_new))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)      # append new key to the cached keys  -> (B, h, t+1, Dk)
            v = torch.cat([cache[1], v], dim=2)      # append new value
        out, _ = scaled_dot_product_attention(q, k, v, mask=None)   # 1 query over all t+1 keys
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.W_O(out), (k, v)                 # return output AND the grown cache

class GPTBlock(GPTBlock):                            # matching step() for the block
    def step(self, x_new, cache):
        a, cache = self.self_attn.step(self.ln1(x_new), cache)
        x_new = x_new + a
        x_new = x_new + self.ffn(self.ln2(x_new))
        return x_new, cache

@torch.no_grad()
def generate_cached(model, ids, n_new):
    B, T = ids.shape
    caches = [None] * len(model.blocks)
    # prefill: feed the prompt one token at a time to seed every layer's cache
    for t in range(T):
        x = model.tok(ids[:, t:t+1]) + model.pos(torch.arange(t, t+1))
        for i, blk in enumerate(model.blocks):
            x, caches[i] = blk.step(x, caches[i])
    last = model.head(model.norm(x))
    # decode: one new token per step, feeding ONLY that token (the cache holds the rest)
    for t in range(T, T + n_new):
        nxt = last[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        x = model.tok(nxt) + model.pos(torch.arange(t, t+1))
        for i, blk in enumerate(model.blocks):
            x, caches[i] = blk.step(x, caches[i])
        last = model.head(model.norm(x))
    return ids

# proof: the cache changes speed, not output
torch.manual_seed(0)
gpt = TinyGPT(vocab=50, D=32, h=4, L=2)
start = torch.randint(0, 50, (1, 3))
assert torch.equal(generate(gpt, start, 6), generate_cached(gpt, start, 6))   # identical tokens
```

(The `class X(X)` trick just tacks the new method onto the class defined earlier so the note reads
top-to-bottom; in real code it is one class body.)

---

## The one bridge worth carrying

The decoder is not a new architecture. It is the encoder's attention with two edits: a triangular
mask so the present cannot see the future, and a second wiring where queries and keys/values come
from different sources. **Attention is attention** — patches, timesteps, nodes, and generated
tokens are all just sequences; the mask decides what may look at what, and the Q vs K/V split
decides which two things are being related. Once that clicks, encoder, decoder, ViT, a graph
transformer, and a time-series forecaster are the same primitive under different masks and wirings.

---

[← Back to contents](index.html) · [Transformer Architecture →](transformers.html)
