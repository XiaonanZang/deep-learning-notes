---
title: Deep Learning Notes
---

# Deep Learning Notes

From-scratch PyTorch implementations and plain-language explanations of the core
building blocks of modern deep learning. Each note pairs the intuition, the math,
and a working, commented implementation you can run.

## Contents

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

**Cold Drills**

Timed, blank-file implementation exercises. Each one pairs a realistic first attempt with a
critique and a corrected, runnable solution — training the gap between *understanding* a
mechanism and *typing it correctly under time pressure*.

- [Multi-Head Attention from Scratch](https://xiaonanzang.github.io/deep-learning-notes/cold-drill-multi-head-attention.html)
  — implement multi-head self-attention and a numerically stable softmax from scratch, with a
  worked critique of the six most common bugs.
- [Multi-Head Attention — Spaced Repetition (Pass 2)](https://xiaonanzang.github.io/deep-learning-notes/cold-drill-multi-head-attention-2.html)
  — the same problem re-attempted cold days later. The design bugs are fixed, but it still does
  not run — a worked example of how the failure *moves* from design to execution precision.

---

*These notes are written to be read in short sittings. Each section stands on its own.*
