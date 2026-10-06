# Changelog

## Unreleased — 0.1.1

First standalone release of `sol-metric`; the import package is `sol_metric`.

- SOL, the double-sliced Wasserstein distance between text corpora, with NumPy
  CPU execution and optional PyTorch CUDA execution.
- Public API: `SOL`, `SOLConfig`, `Features`, `TextEncoder` and the `Encoder`
  protocol, with type annotations (`py.typed`).
- Text encoding with Transformers: Dream 7B by default, with the paper's
  encoders pinned to the paper's checkpoint commits.
- Memory-bounded, streaming feature computation and optional disk caches.
- LM1B and OpenWebText example comparisons, and frozen 5k validation pairs that
  match the paper's evaluation code.

Model weights and research experiments are not included. The source archive
contains the examples, tests and the 5k validation manifest; the frozen 5k text
archives are in the Git repository only. The wheel contains the library only.
