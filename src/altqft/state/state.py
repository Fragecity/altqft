from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit


def _state_amplitude(solutions: Sequence[int]) -> float:
    return 1.0 / math.sqrt(len(solutions))


def _state_vector(
    solutions: Sequence[int],
    n: int,
    *,
    complex_output: bool,
) -> np.ndarray:
    dtype = complex if complex_output else float
    state = np.zeros(1 << n, dtype=dtype)
    state[list(solutions)] = _state_amplitude(solutions)
    return state


def initial_state_from_solutions(solutions: Sequence[int], n: int) -> list[float]:
    return [float(value) for value in _state_vector(solutions, n, complex_output=False)]


def qiskit_initial_state(solutions: Sequence[int], n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.initialize(
        _state_vector(solutions, n, complex_output=True),
        range(n),
    )
    return qc
