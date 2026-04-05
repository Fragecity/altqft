from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from qiskit import QuantumCircuit
from torch import Tensor, nn

from altqft.circuits.layouts import count_required_phases, final_layer, iter_active_layers
from altqft.circuits.ph_generators import ph_1_hlayout, ph_1_parametrized
from altqft.nn.process_qc import (
    _torch_exact_support_indices,
    _torch_probability_vector_from_support,
    _torch_surrogate_support_indices,
)
from altqft.nn.unitary_rows import (
    apply_controlled_phase_rows,
    apply_hadamard_rows,
)

FI_EPSILON = 1e-12
OBJECTIVES = {"min_fi", "shift_ce_mean"}


def _probability_distribution(
    unitary: Tensor,
    period: int,
    shift: int = 0,
    *,
    exact_support: bool = False,
) -> Tensor:
    dimension = unitary.shape[0]
    support_indices = (
        _torch_exact_support_indices(
            dimension,
            period,
            shift,
            device=unitary.device,
        )
        if exact_support
        else _torch_surrogate_support_indices(
            dimension,
            period,
            shift,
            device=unitary.device,
        )
    )
    return _torch_probability_vector_from_support(unitary, support_indices)


def fisher_information(prob1: Tensor, prob2: Tensor, eps: float = FI_EPSILON) -> Tensor:
    denominator = prob1.clamp_min(eps)
    return ((prob1 - prob2).pow(2) / denominator).sum()


def shift_ce_mean_loss_from_distributions(
    shift_distributions: Tensor,
    *,
    eps: float = FI_EPSILON,
) -> tuple[Tensor, Tensor]:
    if shift_distributions.ndim != 2:
        raise ValueError("shift_distributions must have shape (num_shifts, state_space)")
    mean_distribution = shift_distributions.mean(dim=0)
    log_mean_distribution = mean_distribution.clamp_min(eps).log()
    shift_ce = -(shift_distributions * log_mean_distribution.unsqueeze(0)).sum(dim=1)
    shift_l1 = (shift_distributions - mean_distribution.unsqueeze(0)).abs().sum(dim=1)
    return shift_ce.mean(), shift_l1.mean()


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

    def probability_distribution(
        self,
        unitary: Tensor,
        period: int,
        shift: int = 0,
        *,
        exact_support: bool = False,
    ) -> Tensor:
        return _probability_distribution(
            unitary,
            period,
            shift,
            exact_support=exact_support,
        )

    def min_fi(self, period_range: Iterable[int], *, exact_support: bool = False) -> Tensor:
        periods = list(period_range)
        if not periods:
            raise ValueError("period_range must not be empty")

        unitary = self.build_unitary()
        fi_values = [
            fisher_information(
                self.probability_distribution(
                    unitary,
                    period,
                    exact_support=exact_support,
                ),
                self.probability_distribution(
                    unitary,
                    period + 1,
                    exact_support=exact_support,
                ),
            )
            for period in periods
        ]
        return torch.stack(fi_values).amin()

    def shift_ce_mean_loss(
        self,
        period_range: Iterable[int],
        *,
        exact_support: bool = False,
        eps: float = FI_EPSILON,
    ) -> tuple[Tensor, Tensor]:
        periods = list(period_range)
        if not periods:
            raise ValueError("period_range must not be empty")

        unitary = self.build_unitary()
        period_losses: list[Tensor] = []
        period_shift_l1s: list[Tensor] = []
        for period in periods:
            shift_distributions = torch.stack(
                [
                    self.probability_distribution(
                        unitary,
                        period,
                        shift,
                        exact_support=exact_support,
                    )
                    for shift in range(period)
                ]
            )
            period_loss, period_shift_l1 = shift_ce_mean_loss_from_distributions(
                shift_distributions,
                eps=eps,
            )
            period_losses.append(period_loss)
            period_shift_l1s.append(period_shift_l1)

        return torch.stack(period_losses).mean(), torch.stack(period_shift_l1s).mean()

    def forward(self, period_range: Iterable[int]) -> Tensor:
        return self.min_fi(period_range)

    def export_phases(self) -> list[float]:
        return self.phases.detach().cpu().tolist()

    def current_circuit(self) -> QuantumCircuit:
        return ph_1_parametrized(self.nqubit, self.export_phases())
