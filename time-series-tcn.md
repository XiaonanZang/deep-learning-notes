---
title: Time-Series Models: Causal Convolutions, TCN, and Autoregressive Forecasting
---

# Time-Series: Causal Convolutions, TCN, and Autoregressive Forecasting

Forecasting has one hard constraint — **the present may not see the future** — and three different
architectures that all enforce it. This note builds the convolutional one (a causal TCN) from
scratch, but frames it inside the bigger picture: why a sequence model at all, how it connects to
the state-estimation methods you already know, the design space (RNN vs TCN vs Transformer), how you
actually forecast, and where foundation models like Chronos fit.

The arc: **why sequences → from classical state-space models to learned ones → the three sequence
architectures → causal conv + TCN + code → autoregressive forecasting → zero-shot foundation models
→ two real systems → practical gotchas.**

Everything here runs.

---

## 1. Why a sequence model, and the one constraint

A time series is data with a **temporal order**: `x_1, x_2, ..., x_T`. The order *is* the structure
(same "architecture follows data structure" idea as CNN=grid, GNN=graph). Tasks:

- **Forecasting** — predict future values (demand, price, load, a robot's next position).
- **Classification** — label a whole series (is this ECG arrhythmic?).
- **Anomaly detection** — flag points that break the pattern.
- **Imputation** — fill missing steps.

The one constraint that shapes every architecture here: **causality**. At inference the future does
not exist yet, so a forecaster must predict step *t* from `≤ t` only. If any future information leaks
into the prediction of the present, the model cheats in training and collapses in deployment. Every
mechanism below is really *a different way to enforce "only the past."*

---

## 2. From classical state-space models to learned ones

You have likely met sequences as **hand-designed dynamical models**:

| Classical method | What it does | The "intelligence" is... |
|---|---|---|
| **ARIMA / exponential smoothing** | linear forecast from past values | the model form + fitted coefficients you choose |
| **Kalman filter** | optimal state estimate for *linear-Gaussian* dynamics | the transition / observation matrices you specify |
| **HMM** (forward–backward) | posterior over discrete hidden states | the transition / emission probabilities |
| **Particle filter** | state estimate for *nonlinear* dynamics | the motion / observation models you write |

Common thread: **you write down the dynamics, then run a fixed inference algorithm.** A Kalman filter
is *the* optimal estimator — *if* your linear-Gaussian model is correct. When it is not (real
nonlinear, non-Gaussian data), you are stuck hand-crafting extensions.

A **learned sequence model flips it**: instead of specifying the dynamics, it **learns the temporal
pattern from data**. Same shift as CNN vs SIFT, or GNN vs Dijkstra — hand-designed models give way
to learned ones. A TCN, RNN, or Transformer is a Kalman filter whose dynamics were *learned* rather
than assumed, and which is free to be arbitrarily nonlinear.

---

## 3. Three ways to model a sequence

Every deep sequence model is one of three mechanisms, and each enforces causality differently:

| | **RNN / LSTM** | **TCN (causal conv)** | **Transformer (causal attn)** |
|---|---|---|---|
| Mechanism | recurrence (a hidden state rolled forward) | 1-D convolution over time | attention over timesteps |
| Enforces causality via | only past feeds the state | **left padding** (structural) | **`-inf` mask** on scores |
| Training | sequential, `O(T)` steps | **parallel** across time | **parallel** across time |
| Receptive field | in principle unbounded (but vanishing gradients) | fixed, grows with depth × dilation | the whole past, every layer |
| Cost per step (inference) | `O(1)` | `O(k)` | `O(T)` (or `O(1)` with a KV cache) |

This note builds the **TCN**. It is the sweet spot for many problems: parallel to train (unlike
RNNs), cheap and local (unlike attention), with a long memory via dilation. The Transformer route is
covered in the [decoder note](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html).

---

## 4. Causal convolution and the dilated TCN block

**Causal 1-D convolution.** A normal `Conv1d` centred on position *t* reads both sides of *t* — it
leaks the future. Fix it by padding only on the **left**, so the kernel at *t* sees `t, t-1, ...,
t-(k-1)` and nothing after. That left padding is the convolutional equivalent of the decoder's
causal mask.

**Dilation for long memory.** A single causal conv sees only `k` steps back. Dilation skips `d-1`
inputs between taps, so a kernel of size `k` covers `(k-1)·d + 1` steps. Stack layers with dilations
`1, 2, 4, 8, ...` and the receptive field grows **exponentially** with depth while parameters grow
only linearly. That is the TCN's whole trick.

**Residuals**, exactly as in the transformer blocks, let deep dilated stacks train stably.

---

## 5. From-scratch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation                 # left-pad only -> no peek at the future
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
    def forward(self, x):                                  # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))          # pad left, conv -> (B, out_ch, T)

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel, dilation)
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.act(self.c2(y))
        return x + y                                       # residual

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
the model is genuinely causal (nothing after position `t` influences the output at `t`). The output
is a next-step prediction at *every* position — the direct analogue of a language model's
per-position next-token logits.

---

## 6. Autoregressive forecasting

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

Same autoregressive pattern as text generation — the "tokens" are continuous values and the head
regresses a number instead of emitting vocab logits.

---

## 7. Forecasting a series you never trained on: zero-shot foundation models

A classical forecaster (ARIMA, or even a TCN) is fit to **one** series; a new series means a new fit.
The recent shift — and this is the direct parallel to *inductive* GNNs — is **time-series foundation
models** that forecast a series they were **never trained on**, zero-shot.

**Chronos** is the cleanest example, and it is pure recycling of the decoder note: it **tokenises**
the series (quantises continuous values into a fixed vocabulary), then trains a **decoder-only
transformer** to predict the next token — literally the GPT recipe, applied to timesteps instead of
words. Pre-trained on a huge corpus of diverse series, it generalises to an unseen one the way an LLM
generalises to an unseen prompt. The two notes meet here: **a forecaster is a causal sequence model,
whether the causality comes from a conv's padding or a transformer's mask, and whether it is fit to
one series or pre-trained across millions.**

---

## 8. Two systems in practice

**(a) Time-series foundation models (Huzefa's world).**
Chronos and its kin (TimesFM, Moirai) aim to be the "GPT of forecasting": one pre-trained model that
forecasts demand, energy load, web traffic, or sensor data **zero-shot**, no per-series training.
Under the hood it is the decoder-only transformer from these notes, over tokenised values, trained
with next-token prediction. The value proposition is exactly the LLM one — amortise training across
everything, then generalise to the new series for free.

**(b) Trajectory forecasting in autonomous driving (your world).**
An agent's motion *is* a time series: past positions `(x, y)_{t-H..t}` in, future positions
`(x, y)_{t+1..t+F}` out. It is an autoregressive forecast (or a chunked multi-step one) with the same
causal constraint — you may only use the observed past to predict the future. A causal temporal model
(TCN or causal-attention) encodes each agent's history, then rolls it forward. This is the *temporal*
half of motion prediction; the *spatial* half — which agents and lanes interact — is the graph half
from the [GAT note](https://xiaonanzang.github.io/deep-learning-notes/graph-attention.html). Real AV
stacks fuse both: a graph over agents and lanes, a causal sequence model over time.

---

## 9. Two practical gotchas

**Autoregressive error accumulation.** In multi-step forecasting you feed the model *its own*
predictions. A small error at step *t+1* becomes input for *t+2*, so errors **compound** and the
forecast drifts — the same covariate-shift problem as imitation learning (train on ground truth,
test on your own noisy outputs). Mitigations: predict a **chunk** of steps at once (fewer feedback
hops), train with *scheduled sampling* (sometimes feed your own predictions during training), or
forecast a **distribution** and track uncertainty rather than a single path.

**Look-ahead leakage.** The most common and most dangerous time-series bug: accidentally letting
future information into a feature or a normalisation statistic (e.g. scaling by the whole series'
mean, which includes the future). It inflates validation scores and dies in production. Causal
padding exists precisely to make leakage structurally impossible — but leakage can still sneak in
through feature engineering and preprocessing, so audit those against the "only the past" rule too.

---

## The one bridge worth carrying

Forecasting is "predict the next step from the past," and there are three ways to enforce the "from
the past": recurrence (RNN), left-padded convolution (TCN — cheap, local, parallel), or a causal
attention mask (Transformer — flexible, long-range). All three are learned successors to the Kalman
filter: the dynamics are learned, not assumed. Run any of them autoregressively to forecast ahead,
pre-train one across many series and it forecasts a new one zero-shot (Chronos). Timesteps are just
another sequence — the same causal constraint, the same generation loop, a regression head instead
of a vocab head.

---

[← Back to contents](index.html) · [The Transformer Decoder →](transformer-decoder.html)
