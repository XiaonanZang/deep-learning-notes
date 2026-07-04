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
- [Vision Transformers (ViT)](https://xiaonanzang.github.io/deep-learning-notes/vision-transformers.html)
  — treating an image as a sequence of patches and feeding it through a standard
  transformer encoder.

**Cold Drills**

Timed, blank-file implementation exercises. Each one pairs a realistic first attempt with a
critique and a corrected, runnable solution — training the gap between *understanding* a
mechanism and *typing it correctly under time pressure*.

- [Multi-Head Attention from Scratch](https://xiaonanzang.github.io/deep-learning-notes/cold-drill-multi-head-attention.html)
  — implement multi-head self-attention and a numerically stable softmax from scratch, with a
  worked critique of the six most common bugs.

---

*These notes are written to be read in short sittings. Each section stands on its own.*
