from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from altqft.circuits.layouts import alternating_layout, count_required_phases
from altqft.circuits.ph_core import ph_phase, ph_qc


def _uniform_phases(size: int) -> np.ndarray:
    return np.random.uniform(0.0, np.pi, size=size)


def _qft_phases(nqubit: int) -> np.ndarray:
    return np.fromiter(
        (
            np.pi / (2 ** (target - control))
            for control in range(nqubit)
            for target in range(control + 1, nqubit)
        ),
        dtype=float,
        count=nqubit * (nqubit - 1) // 2,
    )


def _random_layout(nqubit: int, nlayer: int) -> list[int]:
    return [int(value) for value in np.random.randint(0, nlayer + 1, size=nqubit)]


def _layout_with_all_layers(nqubit: int, nlayer: int) -> list[int]:
    base = np.arange(nlayer + 1, dtype=int)
    remainder = np.random.randint(0, nlayer + 1, size=nqubit - nlayer - 1)
    hlayout = np.concatenate((base, remainder))
    np.random.shuffle(hlayout)
    return [int(value) for value in hlayout]


def ph_1_hlayout(nqubit: int) -> list[int]:
    return alternating_layout(nqubit)


def qft(nqubit: int) -> QuantumCircuit:
    return ph_qc(list(range(nqubit)), _qft_phases(nqubit))


def ph_1(nqubit: int) -> QuantumCircuit:
    return ph_phase(ph_1_hlayout(nqubit))


def ph_1_parametrized(
    nqubit: int,
    phases: Sequence[float] | np.ndarray,
) -> QuantumCircuit:
    hlayout = ph_1_hlayout(nqubit)
    phase_array = np.asarray(phases, dtype=float)
    expected_phase_count = count_required_phases(hlayout)

    if phase_array.shape != (expected_phase_count,):
        raise ValueError(
            f"ph_1_parametrized expects {expected_phase_count} phases, "
            f"got {phase_array.size}"
        )

    return ph_qc(hlayout, phase_array)


def ph_random(nqubit: int, nlayer: int) -> QuantumCircuit:
    return ph_phase(_random_layout(nqubit, nlayer))


def ph_1_random(nqubit: int) -> QuantumCircuit:
    hlayout = ph_1_hlayout(nqubit)
    return ph_qc(hlayout, _uniform_phases(count_required_phases(hlayout)))


def ph_random_phase(nqubit: int, nlayer: int) -> QuantumCircuit:
    if nlayer >= nqubit:
        raise ValueError("nlayer must be smaller than nqubit")

    hlayout = _layout_with_all_layers(nqubit, nlayer)
    return ph_qc(hlayout, _uniform_phases(count_required_phases(hlayout)))
