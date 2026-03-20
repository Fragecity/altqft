# Alternative QFT (altqft)

This repository contains the source code, numerical experiments, and manuscript files for exploring foundational alternatives to the Quantum Fourier Transform (QFT) for solving the Hidden Subgroup Problem (HSP).


## 📖 Project Overview

While standard HSP solvers rely heavily on the QFT, implementing the full QFT requires long-range controlled-phase operations and deep circuits, which remain challenging for near-term quantum experiments. Instead of treating the QFT as a monolithic target for gate-level approximation, this project isolates the structural properties responsible for extracting hidden subgroup information: **shift invariance** and **Fisher information**.

Key contributions of this work include:
* **Shift-Invariant Circuits**: Deriving necessary conditions (e.g., $U_{ij}=1/\sqrt{|G|}e^{i\theta}$) for circuits to preserve subgroup coset interference patterns.
* **The PH Circuit Family**: Identifying and evaluating a concrete, shallow circuit family that satisfies shift-invariance and serves as a principled alternative to QFT.
* **Information-Theoretic Scaling**: Numerically evaluating how Fisher information scales with qubit count and circuit depth across groups like $\mathbb{Z}_{2^n}$ and finite-dimensional approximations of $\mathbb{Z}_q$.
* **Neural Network Post-Processing**: Demonstrating that the hidden subgroup can be efficiently reconstructed from measurement statistics using a learnable classical decoder.
* **Noise Robustness**: Analyzing the degradation of PH circuits under noisy dynamics compared to standard QFT baselines.

## 📂 Repository Structure

* `src/altqft/`: Core Python package containing quantum circuit constructions, state generators, and neural network models.
* `scripts/`: Executable scripts for running large-scale numerical experiments, Fisher information calculations, and generating figures.
* `tests/`: Unit tests validating circuit unitaries, commutative properties, and neural network dimensions.
* `doc/`: Manuscripts, research notes, and Typst/LaTeX source files.
* `figs/` & `data/`: Generated plots and cached experimental results.

## 🚀 Quick Start

This project recommends using [uv](https://github.com/astral-sh/uv) for high-performance Python package management. 

Ensure that `uv` is installed on your system. Then, clone the repository and build the environment:

```bash
uv sync