from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from altqft.circuits.layouts import alternating_layout, count_required_phases
from altqft.circuits.HPcore import HPphase, HPqc


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


def HP1_hlayout(nqubit: int) -> list[int]:
    return alternating_layout(nqubit)


def qft(nqubit: int) -> QuantumCircuit:
    return HPqc(list(range(nqubit)), _qft_phases(nqubit))


def HP1(nqubit: int) -> QuantumCircuit:
    return HPphase(HP1_hlayout(nqubit))


def HP1_parametrized(
    nqubit: int,
    phases: Sequence[float] | np.ndarray,
) -> QuantumCircuit:
    hlayout = HP1_hlayout(nqubit)
    phase_array = np.asarray(phases, dtype=float)
    expected_phase_count = count_required_phases(hlayout)

    if phase_array.shape != (expected_phase_count,):
        raise ValueError(
            f"HP1_parametrized expects {expected_phase_count} phases, "
            f"got {phase_array.size}"
        )

    return HPqc(hlayout, phase_array)


def _hp1_shared_distance(control: int, target: int, nqubit: int) -> int:
    return min(abs(control - target), control - target + nqubit - 1)


def _hp1_shared_phase_distances(nqubit: int) -> tuple[int, ...]:
    controls = tuple(index for index in range(nqubit) if index % 2 == 0)
    targets = tuple(index for index in range(nqubit) if index % 2 == 1)

    return tuple(
        sorted(
            {
                _hp1_shared_distance(control, target, nqubit)
                for control in controls
                for target in targets
                if _hp1_shared_distance(control, target, nqubit) < nqubit / 3
            }
        )
    )


def HP1_shared_phase_distances(nqubit: int) -> tuple[int, ...]:
    return _hp1_shared_phase_distances(nqubit)


def HP1_shared_phase_count(nqubit: int) -> int:
    return len(HP1_shared_phase_distances(nqubit))


def HP1_shared_parameter(
    nqubit: int,
    phases: Sequence[float] | np.ndarray,
) -> QuantumCircuit:
    controls = tuple(index for index in range(nqubit) if index % 2 == 0)
    targets = tuple(index for index in range(nqubit) if index % 2 == 1)
    phase_distances = _hp1_shared_phase_distances(nqubit)
    distance_to_phase_index = {
        distance: index for index, distance in enumerate(phase_distances)
    }
    phase_array = np.asarray(phases, dtype=float)
    expected_phase_count = len(phase_distances)

    if phase_array.shape != (expected_phase_count,):
        raise ValueError(
            f"HP1_shared_parameter expects {expected_phase_count} phases, "
            f"got {phase_array.size}"
        )

    qc = QuantumCircuit(nqubit)

    for control in controls:
        qc.h(control)
        for target in targets:
            distance = _hp1_shared_distance(control, target, nqubit)
            if distance < nqubit / 3:
                qc.cp(phase_array[distance_to_phase_index[distance]], control, target)

    for target in targets:
        qc.h(target)

    return qc


def HPrandom(nqubit: int, nlayer: int) -> QuantumCircuit:
    return HPphase(_random_layout(nqubit, nlayer))


def HP1_random(nqubit: int) -> QuantumCircuit:
    hlayout = HP1_hlayout(nqubit)
    return HPqc(hlayout, _uniform_phases(count_required_phases(hlayout)))


def HPrandom_phase(nqubit: int, nlayer: int) -> QuantumCircuit:
    if nlayer >= nqubit:
        raise ValueError("nlayer must be smaller than nqubit")

    hlayout = _layout_with_all_layers(nqubit, nlayer)
    return HPqc(hlayout, _uniform_phases(count_required_phases(hlayout)))
