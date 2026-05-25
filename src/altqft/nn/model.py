from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from qiskit import QuantumCircuit
from torch import Tensor, nn

from altqft.circuits.layouts import count_required_phases, final_layer, iter_active_layers
from altqft.circuits.HPgenerators import (
    HP1_hlayout,
    HP1_parametrized,
    HP1_shared_parameter,
    HP1_shared_phase_distances,
)
from altqft.nn.process_qc import (
    _torch_exact_support_indices,
    _torch_probability_vector_from_support,
    _torch_surrogate_support_indices,
)
from altqft.nn.unitary_rows import (
    apply_controlled_phase_rows,
    apply_controlled_phase_state,
    apply_hadamard_rows,
    apply_hadamard_state,
)

FI_EPSILON = 1e-12
OBJECTIVES = {"min_fi", "shift_ce_mean", "hp1_shared_fi_shift"}
ANSATZES = {"HP1", "HP1_shared"}


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


def _periodic_state_vector(
    size: int,
    period: int,
    shift: int,
    *,
    exact_support: bool,
    device: torch.device,
) -> Tensor:
    support_indices = (
        _torch_exact_support_indices(
            size,
            period,
            shift,
            device=device,
        )
        if exact_support
        else _torch_surrogate_support_indices(
            size,
            period,
            shift,
            device=device,
        )
    )
    if support_indices.numel() < 1:
        raise ValueError("support_indices must not be empty")

    state = torch.zeros(size, dtype=torch.complex64, device=device)
    amplitude = torch.tensor(
        1.0 / math.sqrt(float(support_indices.numel())),
        dtype=torch.complex64,
        device=device,
    )
    state.index_fill_(0, support_indices, amplitude)
    return state


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
    log_shift_distribution = shift_distributions.clamp_min(eps).log()
    log_mean_distribution = mean_distribution.clamp_min(eps).log()
    shift_ce = (
        shift_distributions
        * (log_shift_distribution - log_mean_distribution.unsqueeze(0))
    ).sum(dim=1)
    shift_l1 = (shift_distributions - mean_distribution.unsqueeze(0)).abs().sum(dim=1)
    return shift_ce.mean(), shift_l1.mean()


def shift_ce_sum_loss_from_distributions(
    shift_distributions: Tensor,
    *,
    eps: float = FI_EPSILON,
) -> tuple[Tensor, Tensor]:
    if shift_distributions.ndim != 2:
        raise ValueError("shift_distributions must have shape (num_shifts, state_space)")
    mean_distribution = shift_distributions.mean(dim=0)
    log_shift_distribution = shift_distributions.clamp_min(eps).log()
    log_mean_distribution = mean_distribution.clamp_min(eps).log()
    shift_ce = (
        shift_distributions
        * (log_shift_distribution - log_mean_distribution.unsqueeze(0))
    ).sum(dim=1)
    shift_l1 = (shift_distributions - mean_distribution.unsqueeze(0)).abs().sum(dim=1)
    return shift_ce.sum(), shift_l1.mean()


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
        self.hlayout = HP1_hlayout(nqubit)
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

    def _apply_state_layer(
        self,
        state: Tensor,
        controls: Sequence[int],
        targets: Sequence[int],
        start_index: int,
    ) -> tuple[Tensor, int]:
        phase_index = start_index

        for control in controls:
            state = apply_hadamard_state(state, control)
            for target in targets:
                state = apply_controlled_phase_state(
                    state,
                    control,
                    target,
                    self.phases[phase_index],
                )
                phase_index += 1

        return state, phase_index

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

    def evolve_state(self, state: Tensor) -> Tensor:
        phase_index = 0

        for layer in iter_active_layers(self.hlayout):
            state, phase_index = self._apply_state_layer(
                state,
                layer.controls,
                layer.targets,
                phase_index,
            )

        for qubit in final_layer(self.hlayout):
            state = apply_hadamard_state(state, qubit)
        return state

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

    def period_probability_distribution(
        self,
        period: int,
        shift: int = 0,
        *,
        exact_support: bool = False,
    ) -> Tensor:
        state = _periodic_state_vector(
            1 << self.nqubit,
            period,
            shift,
            exact_support=exact_support,
            device=self.phases.device,
        )
        return self.evolve_state(state).abs().pow(2)

    def fi_for_period(self, period: int, *, exact_support: bool = False) -> Tensor:
        return fisher_information(
            self.period_probability_distribution(
                period,
                exact_support=exact_support,
            ),
            self.period_probability_distribution(
                period + 1,
                exact_support=exact_support,
            ),
        )

    def min_fi(self, period_range: Iterable[int], *, exact_support: bool = False) -> Tensor:
        periods = list(period_range)
        if not periods:
            raise ValueError("period_range must not be empty")

        if torch.is_grad_enabled() and self.phases.requires_grad:
            with torch.no_grad():
                detached_values = torch.stack(
                    [
                        self.fi_for_period(period, exact_support=exact_support)
                        for period in periods
                    ]
                )
            best_period = periods[int(detached_values.argmin().detach().cpu().item())]
            return self.fi_for_period(best_period, exact_support=exact_support)

        fi_values = [
            self.fi_for_period(period, exact_support=exact_support)
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

        period_losses: list[Tensor] = []
        period_shift_l1s: list[Tensor] = []
        for period in periods:
            shift_distributions = torch.stack(
                [
                    self.period_probability_distribution(
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
        return HP1_parametrized(self.nqubit, self.export_phases())


class HP1SharedParameterModel(PH1MinFIModel):
    def __init__(self, nqubit: int, init_phases: Sequence[float] | None = None) -> None:
        nn.Module.__init__(self)
        self.nqubit = nqubit
        self.controls = tuple(index for index in range(nqubit) if index % 2 == 0)
        self.targets = tuple(index for index in range(nqubit) if index % 2 == 1)
        self.phase_distances = HP1_shared_phase_distances(nqubit)
        self.distance_to_phase_index = {
            distance: index for index, distance in enumerate(self.phase_distances)
        }
        self.phase_count = len(self.phase_distances)
        self.phases = nn.Parameter(_initial_phase_tensor(self.phase_count, init_phases))

    @staticmethod
    def _distance(control: int, target: int, nqubit: int) -> int:
        return min(abs(control - target), control - target + nqubit - 1)

    def _phase_index(self, control: int, target: int) -> int | None:
        distance = self._distance(control, target, self.nqubit)
        if distance >= self.nqubit / 3:
            return None
        return self.distance_to_phase_index[distance]

    def build_unitary(self) -> Tensor:
        unitary = self._identity()

        for control in self.controls:
            unitary = apply_hadamard_rows(unitary, control)
            for target in self.targets:
                phase_index = self._phase_index(control, target)
                if phase_index is not None:
                    unitary = apply_controlled_phase_rows(
                        unitary,
                        control,
                        target,
                        self.phases[phase_index],
                    )

        for target in self.targets:
            unitary = apply_hadamard_rows(unitary, target)
        return unitary

    def evolve_state(self, state: Tensor) -> Tensor:
        for control in self.controls:
            state = apply_hadamard_state(state, control)
            for target in self.targets:
                phase_index = self._phase_index(control, target)
                if phase_index is not None:
                    state = apply_controlled_phase_state(
                        state,
                        control,
                        target,
                        self.phases[phase_index],
                    )

        for target in self.targets:
            state = apply_hadamard_state(state, target)
        return state

    def sampled_shift_invariance_loss(
        self,
        *,
        period_samples: int,
        shift_samples: int,
        exact_support: bool = False,
        eps: float = FI_EPSILON,
    ) -> tuple[Tensor, Tensor]:
        if period_samples < 1:
            raise ValueError("period_samples must be positive")
        if shift_samples < 1:
            raise ValueError("shift_samples must be positive")

        upper_exclusive = max(3, math.ceil(2 ** (self.nqubit / 2)))
        periods = torch.randint(
            2,
            upper_exclusive,
            (period_samples,),
            device=self.phases.device,
        )
        period_losses: list[Tensor] = []
        period_shift_l1s: list[Tensor] = []

        for period_tensor in periods:
            period = int(period_tensor.detach().cpu().item())
            shifts = torch.randint(
                0,
                period,
                (shift_samples,),
                device=self.phases.device,
            )
            shift_distributions = torch.stack(
                [
                    self.period_probability_distribution(
                        period,
                        int(shift.detach().cpu().item()),
                        exact_support=exact_support,
                    )
                    for shift in shifts
                ]
            )
            period_loss, period_shift_l1 = shift_ce_sum_loss_from_distributions(
                shift_distributions,
                eps=eps,
            )
            period_losses.append(period_loss)
            period_shift_l1s.append(period_shift_l1)

        return torch.stack(period_losses).sum(), torch.stack(period_shift_l1s).mean()

    def current_circuit(self) -> QuantumCircuit:
        return HP1_shared_parameter(self.nqubit, self.export_phases())
