---
title: Deep Learning Notes
---

# Deep Learning Notes

From-scratch PyTorch implementations and plain-language explanations of the core
building blocks of modern deep learning. Each note pairs the intuition, the math,
and a working, commented implementation you can run.

## Go Board — drill rotation (priority order)

Work top-down: **~30 min read, ~15 min blank-file drill, repeat.** Order reflects likelihood
and value; `✅` = a clean ~30-min from-scratch drill, `💬` = understand & explain (too big or
conceptual to code cold). Syntax quick-reference for all of these:
[PyTorch Reference Card § C](https://xiaonanzang.github.io/deep-learning-notes/pytorch-reference-card.html).

| # | Subproblem | Read | Fit |
|---|---|---|---|
| 1 | Stable softmax + **multi-head attention** (the anchor) | [Transformer](https://xiaonanzang.github.io/deep-learning-notes/transformers.html) — Steps 3 & 5 | ✅ |
| 2 | Causal masking / causal self-attention | [Decoder](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html) — §2 | ✅ |
| 3 | **GAT layer** (attention over graph neighbours) | [Graph Attention](https://xiaonanzang.github.io/deep-learning-notes/graph-attention.html) — §5 | ✅ |
| 4 | Causal conv / **TCN block** | [Time-Series / TCN](https://xiaonanzang.github.io/deep-learning-notes/time-series-tcn.html) — §5 | ✅ |
| 5 | **Cross-entropy** & **InfoNCE** from scratch | [Activations/Losses](https://xiaonanzang.github.io/deep-learning-notes/activations-losses-optimizers.html) — §2b | ✅ |
| 6 | Cross-attention (Q ≠ K/V) | [Decoder](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html) — §3 | ✅ |
| 7 | **Training loop** | [Reference Card § B](https://xiaonanzang.github.io/deep-learning-notes/pytorch-reference-card.html) · [Activations/Losses §5](https://xiaonanzang.github.io/deep-learning-notes/activations-losses-optimizers.html) | ✅ |
| 8 | **Custom Dataset / DataLoader** | [Reference Card § A](https://xiaonanzang.github.io/deep-learning-notes/pytorch-reference-card.html) | ✅ |
| 9 | Encoder block (MHA + FFN + LN + residual) | [Transformer](https://xiaonanzang.github.io/deep-learning-notes/transformers.html) — Step 7 | ✅* |
| 10 | ViT patch-embedding front-end | [Vision Transformers](https://xiaonanzang.github.io/deep-learning-notes/vision-transformers.html) | ✅ |
| 11 | Decoder AR loop · KV cache | [Decoder](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html) — §6–7 | 💬 |

`✅*` the encoder block is clean but on the larger side (a full block) — size down to a single
sub-layer if the clock is tight.

## Contents

**Reference**

- [PyTorch Reference Card](https://xiaonanzang.github.io/deep-learning-notes/pytorch-reference-card.html)
  — one-glance card: custom `Dataset`/`DataLoader`, the training loop, the tensor-axis
  cheat-sheet (the "it doesn't run" antidote), and `nn.Parameter` + init.

**Foundations**

- [The PyTorch Module Pattern](https://xiaonanzang.github.io/deep-learning-notes/pytorch-module-pattern.html)
  — the one Python-class pattern every from-scratch model uses: `__init__`, `self`,
  `nn.Parameter`, `forward`, and how `.parameters()` feeds the optimizer.
- [Broadcasting in PyTorch](https://xiaonanzang.github.io/deep-learning-notes/broadcasting.html)
  — the two rules for combining tensors of different shapes, the `keepdim` and
  outer-product bugs, and `expand` vs `repeat`.
- [Activations, Losses, Optimizers, and Schedulers](https://xiaonanzang.github.io/deep-learning-notes/activations-losses-optimizers.html)
  — a training-loop briefing: nonlinearities, cross-entropy and InfoNCE from scratch, and
  the AdamW + warmup + cosine recipe.

**Architectures**

- [Transformer Architecture](https://xiaonanzang.github.io/deep-learning-notes/transformers.html)
  — self-attention, softmax stability, multi-head attention, and the full encoder block,
  built step by step from token embeddings to output.
- [The Transformer Decoder](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html)
  — the generation half: causal masking, cross-attention, the 3-sub-layer decoder layer,
  decoder-only (GPT) vs encoder-decoder, an autoregressive loop, and the KV cache.
- [Vision Transformers (ViT)](https://xiaonanzang.github.io/deep-learning-notes/vision-transformers.html)
  — treating an image as a sequence of patches and feeding it through a standard
  transformer encoder.
- [Graph Attention Networks (GAT)](https://xiaonanzang.github.io/deep-learning-notes/graph-attention.html)
  — self-attention masked by the graph's adjacency matrix: nodes attend to their neighbours,
  with the GCN / GraphSAGE contrast and the bridge back to transformer attention.
- [Time-Series: Causal Convolutions, TCN & Autoregressive Forecasting](https://xiaonanzang.github.io/deep-learning-notes/time-series-tcn.html)
  — enforcing "only the past" with causal padding, dilated TCN blocks for long memory, an
  autoregressive forecast loop, and how Chronos-style foundation models use the decoder recipe.

**Applied Domain — Physical AI & Robot Learning**

- [The Physical AI Data Layer](https://xiaonanzang.github.io/deep-learning-notes/physical-ai-data-layer.html)
  — why data, not architecture, is the bottleneck for robot learning, and what a "data layer for
  physical AI" (curation, validation, policy-fit) actually means.
- [Robot Learning Data](https://xiaonanzang.github.io/deep-learning-notes/robot-learning-data.html)
  — imitation learning and the data side of manipulation: how demonstrations are collected, the
  key datasets (DROID, Ego4D, EgoMimic, MimicLabs), the embodiment gap, and what makes data
  "policy-fit."
- [Auto-labeling & the Evidence Pipeline](https://xiaonanzang.github.io/deep-learning-notes/auto-labeling-evidence-pipeline.html)
  — how foundation models (SAM, DINO, video-text embeddings) draft masks, tracks, and embeddings
  that humans review; vector-DB retrieval as the backbone; and why model output is treated as
  evidence, not automatic ground truth.

---

*These notes are written to be read in short sittings. Each section stands on its own.*
