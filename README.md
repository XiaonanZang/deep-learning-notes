# Deep Learning Notes

From-scratch PyTorch implementations and plain-language explanations of the core
building blocks of modern deep learning. Each note pairs the intuition, the math, and a
working, commented implementation you can run.

## Notes

### Foundations

- **[The PyTorch Module Pattern](pytorch-module-pattern.md)** — the one Python-class
  pattern every from-scratch model uses: `__init__`, `self`, `nn.Parameter`, `forward`,
  and how `.parameters()` feeds the optimizer.
- **[Broadcasting in PyTorch](broadcasting.md)** — the two rules for combining tensors of
  different shapes, the `keepdim` and outer-product bugs, and `expand` vs `repeat`.
- **[Activations, Losses, Optimizers, and Schedulers](activations-losses-optimizers.md)** —
  a training-loop briefing: nonlinearities, cross-entropy and InfoNCE from scratch, and
  the AdamW + warmup + cosine recipe.

### Architectures

- **[Transformer Architecture](transformers.md)** — self-attention, softmax stability,
  multi-head attention, and the full encoder block, built step by step from token
  embeddings to output.
- **[Vision Transformers (ViT)](vision-transformers.md)** — treating an image as a
  sequence of patches and feeding it through a standard transformer encoder.

### Cold Drills

Timed, blank-file implementation exercises — a realistic first attempt, a critique, and a
corrected runnable solution.

- **[Multi-Head Attention from Scratch](cold-drill-multi-head-attention.md)** — implement
  multi-head self-attention and a numerically stable softmax from scratch, with a worked
  critique of the six most common bugs.

## Reading

These notes are published via GitHub Pages for clean reading on any device
(including read-later apps). Each note stands on its own and is meant to be read in a
short sitting.
