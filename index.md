---
title: Deep Learning Notes
---

# Deep Learning Notes

From-scratch PyTorch implementations and plain-language explanations of the core
building blocks of modern deep learning. Each note pairs the intuition, the math,
and a working, commented implementation you can run.

## Contents

**Foundations**

- [The PyTorch Module Pattern](pytorch-module-pattern.html) — the one Python-class
  pattern every from-scratch model uses: `__init__`, `self`, `nn.Parameter`, `forward`,
  and how `.parameters()` feeds the optimizer.

**Architectures**

- [Transformer Architecture](transformers.html) — self-attention, softmax stability,
  multi-head attention, and the full encoder block, built step by step from token
  embeddings to output.
- [Vision Transformers (ViT)](vision-transformers.html) — treating an image as a
  sequence of patches and feeding it through a standard transformer encoder.

---

*These notes are written to be read in short sittings. Each section stands on its own.*
