---
title: Time-Series Models: Causal Convolutions, TCN, and Autoregressive Forecasting
---

# Time-Series: Causal Convolutions, TCN, and Autoregressive Forecasting

Forecasting has the same hard constraint as text generation: **the present may not see the
future.** A transformer enforces that with a causal *mask*; a convolutional model enforces it with
causal *padding*. This note builds a causal 1D convolution, stacks it into a dilated **TCN**
(Temporal Convolutional Network) for a long memory, and runs an **autoregressive** forecast — then
connects all of it back to the attention notes and to foundation-model forecasters like Chronos.

Everything here runs.

---

## 1. Causal 1D convolution

A plain `Conv1d` centred on position *t* reads inputs on **both** sides of *t* — which leaks the
future into the prediction. A **causal** convolution fixes this by padding only on the **left**, so
the kernel at position *t* sees positions `t, t-1, ..., t-(k-1)` and nothing after *t*.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation          # how many left-pad zeros keep it causal
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
    def forward(self, x):                            # x: (B, C, T)  channels-first, T is time
        x = F.pad(x, (self.pad, 0))                  # pad ONLY on the left (past side)
        return self.conv(x)                          # (B, out_ch, T) — output length unchanged
```

Padding on the left by exactly `(k-1)·dilation` means the convolution produces an output of the
same length `T`, and every output position `t` is a function of inputs `≤ t` only. That "≤ t" is
the convolutional equivalent of the decoder's causal mask.

---

## 2. Dilated convolutions and the TCN block

A single causal conv only sees `k` steps back. To model long histories without huge kernels or
deep stacks, TCNs use **dilation**: skip `d-1` inputs between taps, so a kernel of size `k` covers
`(k-1)·d + 1` steps. Stack layers with dilations `1, 2, 4, 8, ...` and the receptive field grows
**exponentially** with depth while parameters grow only linearly.

```python
class TCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel, dilation)
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.act(self.c2(y))
        return x + y                                 # residual, exactly like a transformer sub-layer
```

The residual connection is the same trick as the encoder/decoder blocks: it lets gradients skip the
sub-layer so deep dilated stacks train stably.

---

## 3. A from-scratch TCN forecaster

Stack a few blocks with growing dilation, project in and out with 1×1 convs (a 1×1 causal conv is
just a per-timestep linear layer). The model outputs, at **every** position, a prediction of the
next value — the direct analogue of a language model's per-position next-token logits.

```python
class TCNForecaster(nn.Module):
    def __init__(self, ch=16, kernel=3, n_blocks=3):
        super().__init__()
        self.inp = CausalConv1d(1, ch, 1)                                  # 1 series channel -> ch
        self.blocks = nn.ModuleList(
            [TCNBlock(ch, kernel, dilation=2 ** i) for i in range(n_blocks)])  # dilations 1,2,4
        self.head = CausalConv1d(ch, 1, 1)                                 # ch -> 1 predicted value
    def forward(self, x):                            # x: (B, 1, T)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)                          # (B, 1, T): next-step prediction at each position

m = TCNForecaster(ch=16, kernel=3, n_blocks=3)
x = torch.randn(1, 1, 20)
print(m(x).shape)                                    # torch.Size([1, 1, 20])

# causality check: perturb the LAST input; every EARLIER output must be unchanged
x2 = x.clone(); x2[0, 0, -1] += 100.0
print(torch.allclose(m(x)[..., :-1], m(x2)[..., :-1]))   # True — the future never leaked backward
```

Verified: changing the input at the last timestep leaves all earlier outputs bit-identical — proof
the model is genuinely causal (nothing after position `t` influences the output at `t`).

---

## 4. Autoregressive forecasting

To forecast multiple steps ahead, do exactly what the decoder's `generate()` loop does: predict the
next value, append it, feed the longer series back in, repeat.

```python
@torch.no_grad()
def forecast(model, seed, n_steps):                  # seed: (1, 1, T)
    s = seed
    for _ in range(n_steps):
        nxt = model(s)[:, :, -1:]                    # take the LAST position's prediction, (1,1,1)
        s = torch.cat([s, nxt], dim=2)               # append it and roll forward
    return s

print(forecast(m, x, n_steps=5).shape)               # torch.Size([1, 1, 25]): 20 seen + 5 forecast
```

This is the same autoregressive pattern as text generation — the only difference is that the
"tokens" are continuous values and the head regresses a number instead of emitting vocab logits.

---

## 5. The bridge: two ways to enforce "only the past"

Causal conv and causal attention solve the identical problem with different mechanisms:

| | Causal convolution (TCN) | Causal self-attention (decoder) |
|---|---|---|
| How the future is blocked | **left padding** (structural) | **`-inf` mask** on the scores |
| Receptive field | fixed, grows with depth × dilation | the whole past at every layer |
| Cost per step | `O(k)` per position, cheap | `O(T)` per position (or `O(1)` with a KV cache) |
| Long-range dependencies | needs enough dilated depth | direct, any distance in one layer |
| Inductive bias | strong locality (good with little data) | none (needs more data), but more flexible |

Neither is strictly better — it is the same **locality-vs-flexibility / bias-vs-data** trade-off as
CNNs vs ViT in the [ViT note](https://xiaonanzang.github.io/deep-learning-notes/vision-transformers.html).

**Where foundation models sit.** Modern time-series foundation models (e.g. **Chronos**) mostly go
the *attention* route: tokenise the series (Chronos quantises values into a vocabulary) and train a
**decoder-only transformer** to predict the next token — literally the GPT recipe from the decoder
note, applied to timesteps instead of words. So the two notes meet here: a forecaster is a causal
sequence model, whether the causality comes from a conv's padding or a transformer's mask.

---

## The one bridge worth carrying

Forecasting = "predict the next step from the past." Enforce the "from the past" with left-padded
convolutions (a TCN, cheap and local) or a causal attention mask (a decoder-only transformer,
flexible and long-range), then run it autoregressively. Timesteps are just another sequence — the
same generation loop as the decoder, the same causality constraint, a regression head instead of a
vocab head.

---

[← Back to contents](index.html) · [The Transformer Decoder →](transformer-decoder.html)
