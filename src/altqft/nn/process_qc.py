from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

FloatArray = NDArray[np.float64]
ProbFunc = Callable[[int, int], float]
EPSILON = 1e-12


def _probability_vector(unitary: np.ndarray, period: int, shift: int) -> FloatArray:
    row_count = unitary.shape[0] // period
    row_indices = shift + np.arange(row_count) * period
    amplitudes = unitary[row_indices].sum(axis=0)
    return np.asarray(np.abs(amplitudes) ** 2 / row_count, dtype=np.float64)


def _distribution(probability: ProbFunc, size: int, shift: int = 0) -> FloatArray:
    return np.fromiter(
        (probability(column, shift) for column in range(size)),
        dtype=float,
        count=size,
    )


def _fisher_for_period(circuit: QuantumCircuit, period: int, size: int) -> float:
    return fi(
        _distribution(make_prob(circuit, period), size),
        _distribution(make_prob(circuit, period + 1), size),
    )


def make_prob(circuit: QuantumCircuit, period: int) -> ProbFunc:
    unitary = np.asarray(Operator(circuit).data)
    cache: dict[int, FloatArray] = {}

    def prob(col: int, shift: int) -> float:
        values = cache.setdefault(shift, _probability_vector(unitary, period, shift))
        return float(values[col])

    return prob


def fi(prob1: Sequence[float] | FloatArray, prob2: Sequence[float] | FloatArray) -> float:
    first = np.asarray(prob1, dtype=float)
    second = np.asarray(prob2, dtype=float)
    mask = first > EPSILON
    return float(np.sum(((first[mask] - second[mask]) ** 2) / first[mask]))


def min_fi(circuit: QuantumCircuit, period_range: Iterable[int]) -> float:
    size = 1 << circuit.num_qubits
    return min(_fisher_for_period(circuit, period, size) for period in period_range)
