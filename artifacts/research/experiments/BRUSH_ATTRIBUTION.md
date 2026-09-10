# Brush attribution

The projection, spherical harmonic evaluation and alpha compositing in `gaussian_depth_renderer.py` are reimplemented from [Brush](https://github.com/ArthurBrussee/brush), by Arthur Brussee and contributors.
The inspected version is v0.3.0, commit `3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486`.
The corresponding source is in [`crates/brush-render/src/shaders`](https://github.com/ArthurBrussee/brush/tree/3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486/crates/brush-render/src/shaders).
Brush is distributed under the Apache License 2.0, whose text is included in this repository's [LICENSE.txt](../../../LICENSE.txt).

This project's implementation uses NumPy projection and tiled PyTorch compositing on CPU or MPS.
It additionally computes depth moments, center-ordered depth quantiles and conditional Gaussian depth diagnostics.
Those geometry diagnostics are project experiments and are not represented as upstream Brush features or accuracy guarantees.
