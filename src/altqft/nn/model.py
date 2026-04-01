from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from qiskit import QuantumCircuit
from torch import Tensor, nn

from altqft.circuits.layouts import count_required_phases, final_layer, iter_active_layers
from altqft.circuits.ph_generators import ph_1_hlayout, ph_1_parametrized
from altqft.nn.unitary_rows import (
    apply_controlled_phase_rows,
    apply_hadamard_rows,
)

FI_EPSILON = 1e-12


def _probability_distribution(unitary: Tensor, period: int, shift: int = 0) -> Tensor:
    dimension = unitary.shape[0]
    num_k = dimension // period
    row_indices = shift + torch.arange(num_k, device=unitary.device) * period
    selected_rows = unitary.index_select(0, row_indices)
    amplitudes = selected_rows.sum(dim=0)
    return amplitudes.abs().pow(2) / float(num_k)


def fisher_information(prob1: Tensor, prob2: Tensor, eps: float = FI_EPSILON) -> Tensor:
    denominator = prob1.clamp_min(eps)
    return ((prob1 - prob2).pow(2) / denominator).sum()


def _initial_phase_tensor(
    phase_count: int,
    init_phases: Sequence[float] | None,
) -> Tensor:
    if init_phases is None:
        return 2 * math.pi * torch.rand(phase_count, dtype=torch.float32)

    if len(init_phases) != phase_count:
        raise ValueError(
            f"expected {phase_count} initial phases, got {len(init_phases)}"
        )

    return torch.tensor(init_phases, dtype=torch.float32)


class PH1MinFIModel(nn.Module):
    def __init__(self, nqubit: int, init_phases: Sequence[float] | None = None) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.hlayout = ph_1_hlayout(nqubit)
        self.phase_count = count_required_phases(self.hlayout)
        self.phases = nn.Parameter(_initial_phase_tensor(self.phase_count, init_phases))

    def _identity(self) -> Tensor:
        size = 1 << self.nqubit
        return torch.eye(size, dtype=torch.complex64, device=self.phases.device)

    def _apply_layer(
        self,
        unitary: Tensor,
        controls: Sequence[int],
        targets: Sequence[int],
        start_index: int,
    ) -> tuple[Tensor, int]:
        phase_index = start_index

        for control in controls:
            unitary = apply_hadamard_rows(unitary, control)
            for target in targets:
                unitary = apply_controlled_phase_rows(
                    unitary,
                    control,
                    target,
                    self.phases[phase_index],
                )
                phase_index += 1

        return unitary, phase_index

    def build_unitary(self) -> Tensor:
        unitary = self._identity()
        phase_index = 0

        for layer in iter_active_layers(self.hlayout):
            unitary, phase_index = self._apply_layer(
                unitary,
                layer.controls,
                layer.targets,
                phase_index,
            )

        for qubit in final_layer(self.hlayout):
            unitary = apply_hadamard_rows(unitary, qubit)
        return unitary

    def min_fi(self, period_range: Iterable[int]) -> Tensor:
        periods = list(period_range)
        if not periods:
            raise ValueError("period_range must not be empty")

        unitary = self.build_unitary()
        fi_values = [
            fisher_information(
                _probability_distribution(unitary, period),
                _probability_distribution(unitary, period + 1),
            )
            for period in periods
        ]
        return torch.stack(fi_values).amin()

    def forward(self, period_range: Iterable[int]) -> Tensor:
        return self.min_fi(period_range)

    def export_phases(self) -> list[float]:
        return self.phases.detach().cpu().tolist()

    def current_circuit(self) -> QuantumCircuit:
        return ph_1_parametrized(self.nqubit, self.export_phases())
