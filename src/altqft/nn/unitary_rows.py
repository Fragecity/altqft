from __future__ import annotations

import math

import torch
from torch import Tensor

PhaseAngle = Tensor | float


def all_rows(size: int, device: torch.device) -> Tensor:
    return torch.arange(size, device=device, dtype=torch.long)


def paired_row_indices(
    size: int,
    qubit: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    rows = all_rows(size, device)
    lower_rows = rows[((rows >> qubit) & 1) == 0]
    upper_rows = lower_rows | (1 << qubit)
    return lower_rows, upper_rows


def phase_row_indices(
    size: int,
    control: int,
    target: int,
    device: torch.device,
) -> Tensor:
    rows = all_rows(size, device)
    control_mask = ((rows >> control) & 1) == 1
    target_mask = ((rows >> target) & 1) == 1
    return rows[control_mask & target_mask]


def apply_hadamard_rows(unitary: Tensor, qubit: int) -> Tensor:
    lower_rows, upper_rows = paired_row_indices(
        unitary.shape[0],
        qubit,
        unitary.device,
    )
    lower_values = unitary.index_select(0, lower_rows)
    upper_values = unitary.index_select(0, upper_rows)
    scale = 1.0 / math.sqrt(2.0)

    updated = unitary.clone()
    updated.index_copy_(0, lower_rows, (lower_values + upper_values) * scale)
    updated.index_copy_(0, upper_rows, (lower_values - upper_values) * scale)
    return updated


def _phase_factor(theta: PhaseAngle, device: torch.device) -> Tensor:
    if isinstance(theta, Tensor):
        angle = theta.to(device=device, dtype=torch.complex64)
    else:
        angle = torch.tensor(theta, dtype=torch.complex64, device=device)
    return torch.exp(1j * angle)


def apply_controlled_phase_rows(
    unitary: Tensor,
    control: int,
    target: int,
    theta: PhaseAngle,
) -> Tensor:
    active_rows = phase_row_indices(
        unitary.shape[0],
        control,
        target,
        unitary.device,
    )
    if active_rows.numel() == 0:
        return unitary

    updated = unitary.clone()
    active_values = unitary.index_select(0, active_rows)
    updated.index_copy_(
        0,
        active_rows,
        active_values * _phase_factor(theta, unitary.device),
    )
    return updated
