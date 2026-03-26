from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from torch import Tensor

from altqft.circuits.layouts import (
    count_required_phases,
    final_layer,
    iter_active_layers,
)

PhaseInput = Sequence[float] | np.ndarray | Tensor


def _phase_value(phases: PhaseInput, index: int) -> float:
    return float(phases[index])


def _apply_layer(
    qc: QuantumCircuit,
    controls: tuple[int, ...],
    targets: tuple[int, ...],
    phases: PhaseInput,
    start_index: int,
) -> int:
    phase_index = start_index

    for control in controls:
        qc.h(control)
        for target in targets:
            qc.cp(_phase_value(phases, phase_index), control, target)
            phase_index += 1

    return phase_index


def ph_qc(hlayout: Sequence[int], phase: PhaseInput) -> QuantumCircuit:
    qc = QuantumCircuit(len(hlayout))
    phase_index = 0

    for layer in iter_active_layers(hlayout):
        phase_index = _apply_layer(
            qc,
            layer.controls,
            layer.targets,
            phase,
            phase_index,
        )

    for qubit in final_layer(hlayout):
        qc.h(qubit)

    return qc


def ph_phase(hlayout: Sequence[int]) -> QuantumCircuit:
    phases = np.fromiter(
        (
            np.pi / (2 ** abs(target - control))
            for layer in iter_active_layers(hlayout)
            for control in layer.controls
            for target in layer.targets
        ),
        dtype=float,
        count=count_required_phases(hlayout),
    )
    return ph_qc(hlayout, phases)
