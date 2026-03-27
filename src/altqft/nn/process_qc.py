from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from torch import Tensor

FloatArray = NDArray[np.float64]
ProbFunc = Callable[[int, int], float]
EPSILON = 1e-12
SUPPORTED_TORCH_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _probability_vector(unitary: np.ndarray, period: int, shift: int) -> FloatArray:
    row_count = unitary.shape[0] // period
    row_indices = shift + np.arange(row_count) * period
    amplitudes = unitary[row_indices].sum(axis=0)
    return np.asarray(np.abs(amplitudes) ** 2 / row_count, dtype=np.float64)


def resolve_compute_device(device: str = "auto") -> str:
    normalized = device.lower()
    if normalized not in SUPPORTED_TORCH_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_TORCH_DEVICES))
        raise ValueError(f"unsupported device '{device}', expected one of: {supported}")

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but no CUDA device is available")

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("mps requested but no MPS device is available")

    return normalized


def available_cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _torch_probability_vector(unitary: Tensor, period: int, shift: int) -> Tensor:
    row_count = unitary.shape[0] // period
    row_indices = shift + torch.arange(
        row_count,
        device=unitary.device,
        dtype=torch.long,
    ) * period
    amplitudes = unitary.index_select(0, row_indices).sum(dim=0)
    return amplitudes.abs().pow(2) / float(row_count)


def _torch_fi(prob1: Tensor, prob2: Tensor) -> Tensor:
    denominator = prob1.clamp_min(EPSILON)
    return ((prob1 - prob2).pow(2) / denominator).sum()


def _all_rows(size: int, device: torch.device) -> Tensor:
    return torch.arange(size, device=device, dtype=torch.long)


def _paired_row_indices(size: int, qubit: int, device: torch.device) -> tuple[Tensor, Tensor]:
    rows = _all_rows(size, device)
    lower_rows = rows[((rows >> qubit) & 1) == 0]
    upper_rows = lower_rows | (1 << qubit)
    return lower_rows, upper_rows


def _phase_row_indices(
    size: int,
    control: int,
    target: int,
    device: torch.device,
) -> Tensor:
    rows = _all_rows(size, device)
    control_mask = ((rows >> control) & 1) == 1
    target_mask = ((rows >> target) & 1) == 1
    return rows[control_mask & target_mask]


def _apply_hadamard_rows(unitary: Tensor, qubit: int) -> Tensor:
    lower_rows, upper_rows = _paired_row_indices(unitary.shape[0], qubit, unitary.device)
    lower_values = unitary.index_select(0, lower_rows)
    upper_values = unitary.index_select(0, upper_rows)
    scale = 1.0 / math.sqrt(2.0)
    unitary.index_copy_(0, lower_rows, (lower_values + upper_values) * scale)
    unitary.index_copy_(0, upper_rows, (lower_values - upper_values) * scale)
    return unitary


def _apply_controlled_phase_rows(
    unitary: Tensor,
    control: int,
    target: int,
    theta: float,
) -> Tensor:
    active_rows = _phase_row_indices(unitary.shape[0], control, target, unitary.device)
    if active_rows.numel() == 0:
        return unitary

    phase = torch.exp(
        1j * torch.tensor(theta, dtype=torch.float32, device=unitary.device)
    ).to(torch.complex64)
    active_values = unitary.index_select(0, active_rows)
    unitary.index_copy_(0, active_rows, active_values * phase)
    return unitary


def _qubit_index(circuit: QuantumCircuit, qubit: Any) -> int:
    return int(circuit.find_bit(qubit).index)


def _fallback_torch_unitary(circuit: QuantumCircuit, device: torch.device) -> Tensor:
    return torch.tensor(
        np.asarray(Operator(circuit).data),
        dtype=torch.complex64,
        device=device,
    )


def _torch_unitary(circuit: QuantumCircuit, device_name: str) -> Tensor:
    device = torch.device(device_name)
    size = 1 << circuit.num_qubits
    unitary = torch.eye(size, dtype=torch.complex64, device=device)

    for instruction in circuit.data:
        operation = instruction.operation
        op_name = operation.name.lower()

        if op_name == "h":
            qubit = _qubit_index(circuit, instruction.qubits[0])
            unitary = _apply_hadamard_rows(unitary, qubit)
            continue

        if op_name == "cp":
            control = _qubit_index(circuit, instruction.qubits[0])
            target = _qubit_index(circuit, instruction.qubits[1])
            theta = float(operation.params[0])
            unitary = _apply_controlled_phase_rows(unitary, control, target, theta)
            continue

        return _fallback_torch_unitary(circuit, device)

    return unitary


def _fisher_for_period_torch(circuit: QuantumCircuit, period: int, device_name: str) -> float:
    unitary = _torch_unitary(circuit, device_name)
    first = _torch_probability_vector(unitary, period, 0)
    second = _torch_probability_vector(unitary, period + 1, 0)
    return float(_torch_fi(first, second).item())


def _min_fi_torch(circuit: QuantumCircuit, period_range: Iterable[int], device_name: str) -> float:
    periods = tuple(period_range)
    if not periods:
        raise ValueError("period_range must not be empty")

    unitary = _torch_unitary(circuit, device_name)
    fi_values = [
        _torch_fi(
            _torch_probability_vector(unitary, period, 0),
            _torch_probability_vector(unitary, period + 1, 0),
        )
        for period in periods
    ]
    return float(torch.stack(fi_values).amin().item())


def probability_distribution(probability: ProbFunc, size: int, shift: int = 0) -> FloatArray:
    return np.fromiter(
        (probability(column, shift) for column in range(size)),
        dtype=float,
        count=size,
    )


def _fisher_for_period(circuit: QuantumCircuit, period: int, size: int) -> float:
    return fi(
        probability_distribution(make_prob(circuit, period), size),
        probability_distribution(make_prob(circuit, period + 1), size),
    )


def make_prob(circuit: QuantumCircuit, period: int) -> ProbFunc:
    unitary = np.asarray(Operator(circuit).data)
    cache: dict[int, FloatArray] = {}

    def prob(col: int, shift: int) -> float:
        values = cache.setdefault(shift, _probability_vector(unitary, period, shift))
        return float(values[col])

    return prob


def circuit_probability_distribution(
    circuit: QuantumCircuit,
    period: int,
    shift: int = 0,
) -> FloatArray:
    size = 1 << circuit.num_qubits
    return probability_distribution(make_prob(circuit, period), size, shift=shift)


def fi(prob1: Sequence[float] | FloatArray, prob2: Sequence[float] | FloatArray) -> float:
    first = np.asarray(prob1, dtype=float)
    second = np.asarray(prob2, dtype=float)
    mask = first > EPSILON
    return float(np.sum(((first[mask] - second[mask]) ** 2) / first[mask]))


def min_fi(
    circuit: QuantumCircuit,
    period_range: Iterable[int],
    device: str | None = None,
) -> float:
    if device is not None:
        resolved_device = resolve_compute_device(device)
        return _min_fi_torch(circuit, period_range, resolved_device)

    size = 1 << circuit.num_qubits
    return min(_fisher_for_period(circuit, period, size) for period in period_range)
