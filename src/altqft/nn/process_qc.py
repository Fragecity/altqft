from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from torch import Tensor

from altqft.nn.devices import (
    available_cuda_device_count as available_cuda_device_count,
    resolve_compute_device as resolve_compute_device,
)
from altqft.nn.unitary_rows import (
    apply_controlled_phase_rows,
    apply_controlled_phase_state,
    apply_controlled_phase_state_batch,
    apply_hadamard_rows,
    apply_hadamard_state,
    apply_hadamard_state_batch,
)

FloatArray = NDArray[np.float64]
ProbFunc = Callable[[int, int], float]
EPSILON = 1e-12
UNITARY_MIN_FI_MAX_QUBITS = 12
STATE_BATCH_MAX_BYTES = 768 * 1024 * 1024


def _surrogate_support_indices(size: int, period: int, shift: int) -> NDArray[np.int64]:
    support_count = size // period
    return shift + np.arange(support_count, dtype=np.int64) * period


def _exact_support_indices(size: int, period: int, shift: int) -> NDArray[np.int64]:
    return np.arange(shift, size, period, dtype=np.int64)


def _probability_vector_from_support(unitary: np.ndarray, support_indices: NDArray[np.int64]) -> FloatArray:
    if support_indices.size < 1:
        raise ValueError("support_indices must not be empty")
    amplitudes = unitary[:, support_indices].sum(axis=1)
    return np.asarray(np.abs(amplitudes) ** 2 / float(support_indices.size), dtype=np.float64)


def _probability_vector(unitary: np.ndarray, period: int, shift: int) -> FloatArray:
    support_indices = _surrogate_support_indices(unitary.shape[1], period, shift)
    return _probability_vector_from_support(unitary, support_indices)


def _torch_surrogate_support_indices(
    size: int,
    period: int,
    shift: int,
    *,
    device: torch.device,
) -> Tensor:
    support_count = size // period
    return shift + torch.arange(
        support_count,
        device=device,
        dtype=torch.long,
    ) * period


def _torch_exact_support_indices(
    size: int,
    period: int,
    shift: int,
    *,
    device: torch.device,
) -> Tensor:
    return torch.arange(shift, size, period, device=device, dtype=torch.long)


def _torch_probability_vector_from_support(unitary: Tensor, support_indices: Tensor) -> Tensor:
    if support_indices.numel() < 1:
        raise ValueError("support_indices must not be empty")
    amplitudes = unitary.index_select(1, support_indices).sum(dim=1)
    return amplitudes.abs().pow(2) / float(support_indices.numel())


def _torch_probability_vector(unitary: Tensor, period: int, shift: int) -> Tensor:
    support_indices = _torch_surrogate_support_indices(
        unitary.shape[1],
        period,
        shift,
        device=unitary.device,
    )
    return _torch_probability_vector_from_support(unitary, support_indices)


def _torch_periodic_state_vector(
    size: int,
    period: int,
    shift: int,
    *,
    exact_support: bool,
    device: torch.device,
) -> Tensor:
    support_indices = (
        _torch_exact_support_indices(size, period, shift, device=device)
        if exact_support
        else _torch_surrogate_support_indices(size, period, shift, device=device)
    )
    if support_indices.numel() < 1:
        raise ValueError("support_indices must not be empty")

    state = torch.zeros(size, dtype=torch.complex64, device=device)
    state.index_fill_(
        0,
        support_indices,
        torch.tensor(
            1.0 / np.sqrt(float(support_indices.numel())),
            dtype=torch.complex64,
            device=device,
        ),
    )
    return state


def _torch_periodic_state_vectors(
    size: int,
    periods: Sequence[int],
    shift: int,
    *,
    exact_support: bool,
    device: torch.device,
) -> Tensor:
    states = torch.zeros((len(periods), size), dtype=torch.complex64, device=device)
    for row, period in enumerate(periods):
        support_indices = (
            _torch_exact_support_indices(size, period, shift, device=device)
            if exact_support
            else _torch_surrogate_support_indices(size, period, shift, device=device)
        )
        if support_indices.numel() < 1:
            raise ValueError("support_indices must not be empty")
        states[row].index_fill_(
            0,
            support_indices,
            torch.tensor(
                1.0 / np.sqrt(float(support_indices.numel())),
                dtype=torch.complex64,
                device=device,
            ),
        )
    return states


def _fallback_apply_circuit_state(circuit: QuantumCircuit, state: Tensor) -> Tensor:
    unitary = torch.tensor(
        np.asarray(Operator(circuit).data),
        dtype=torch.complex64,
        device=state.device,
    )
    return torch.matmul(unitary, state)


def _fallback_apply_circuit_state_batch(circuit: QuantumCircuit, states: Tensor) -> Tensor:
    unitary = torch.tensor(
        np.asarray(Operator(circuit).data),
        dtype=torch.complex64,
        device=states.device,
    )
    return torch.matmul(states, unitary.T)


def _apply_circuit_state(circuit: QuantumCircuit, state: Tensor) -> Tensor:
    original_state = state
    for instruction in circuit.data:
        operation = instruction.operation
        op_name = operation.name.lower()

        if op_name == "h":
            qubit = _qubit_index(circuit, instruction.qubits[0])
            state = apply_hadamard_state(state, qubit)
            continue

        if op_name == "cp":
            control = _qubit_index(circuit, instruction.qubits[0])
            target = _qubit_index(circuit, instruction.qubits[1])
            theta = float(operation.params[0])
            state = apply_controlled_phase_state(state, control, target, theta)
            continue

        return _fallback_apply_circuit_state(circuit, original_state)

    return state


def _apply_circuit_state_batch(circuit: QuantumCircuit, states: Tensor) -> Tensor:
    original_states = states
    for instruction in circuit.data:
        operation = instruction.operation
        op_name = operation.name.lower()

        if op_name == "h":
            qubit = _qubit_index(circuit, instruction.qubits[0])
            states = apply_hadamard_state_batch(states, qubit)
            continue

        if op_name == "cp":
            control = _qubit_index(circuit, instruction.qubits[0])
            target = _qubit_index(circuit, instruction.qubits[1])
            theta = float(operation.params[0])
            states = apply_controlled_phase_state_batch(states, control, target, theta)
            continue

        return _fallback_apply_circuit_state_batch(circuit, original_states)

    return states


def _torch_circuit_probability_vector(
    circuit: QuantumCircuit,
    period: int,
    shift: int,
    *,
    exact_support: bool = False,
    device: torch.device,
) -> Tensor:
    state = _torch_periodic_state_vector(
        1 << circuit.num_qubits,
        period,
        shift,
        exact_support=exact_support,
        device=device,
    )
    return _apply_circuit_state(circuit, state).abs().pow(2)


def _torch_circuit_probability_vectors(
    circuit: QuantumCircuit,
    periods: Sequence[int],
    shift: int,
    *,
    exact_support: bool = False,
    device: torch.device,
) -> Tensor:
    states = _torch_periodic_state_vectors(
        1 << circuit.num_qubits,
        periods,
        shift,
        exact_support=exact_support,
        device=device,
    )
    return _apply_circuit_state_batch(circuit, states).abs().pow(2)


def _torch_fi(prob1: Tensor, prob2: Tensor) -> Tensor:
    denominator = prob1.clamp_min(EPSILON)
    return ((prob1 - prob2).pow(2) / denominator).sum()


def _torch_fi_batch(prob1: Tensor, prob2: Tensor) -> Tensor:
    denominator = prob1.clamp_min(EPSILON)
    return ((prob1 - prob2).pow(2) / denominator).sum(dim=1)


def _state_batch_period_chunk_size(size: int) -> int:
    bytes_per_state = size * torch.tensor([], dtype=torch.complex64).element_size()
    max_vectors = max(2, STATE_BATCH_MAX_BYTES // bytes_per_state)
    return max(1, int(max_vectors // 2))


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
            unitary = apply_hadamard_rows(unitary, qubit)
            continue

        if op_name == "cp":
            control = _qubit_index(circuit, instruction.qubits[0])
            target = _qubit_index(circuit, instruction.qubits[1])
            theta = float(operation.params[0])
            unitary = apply_controlled_phase_rows(unitary, control, target, theta)
            continue

        return _fallback_torch_unitary(circuit, device)

    return unitary


def _min_fi_torch(circuit: QuantumCircuit, period_range: Iterable[int], device_name: str) -> float:
    periods = tuple(period_range)
    if not periods:
        raise ValueError("period_range must not be empty")

    device = torch.device(device_name)
    if circuit.num_qubits <= UNITARY_MIN_FI_MAX_QUBITS:
        unitary = _torch_unitary(circuit, device_name)
        fi_values = [
            _torch_fi(
                _torch_probability_vector(unitary, period, 0),
                _torch_probability_vector(unitary, period + 1, 0),
            )
            for period in periods
        ]
        return float(torch.stack(fi_values).amin().item())

    size = 1 << circuit.num_qubits
    period_chunk_size = _state_batch_period_chunk_size(size)
    fi_chunks: list[Tensor] = []
    for start in range(0, len(periods), period_chunk_size):
        period_chunk = periods[start : start + period_chunk_size]
        comparison_periods = period_chunk + tuple(period + 1 for period in period_chunk)
        probabilities = _torch_circuit_probability_vectors(
            circuit,
            comparison_periods,
            0,
            device=device,
        )
        first_probs = probabilities[: len(period_chunk)]
        second_probs = probabilities[len(period_chunk) :]
        fi_chunks.append(_torch_fi_batch(first_probs, second_probs))

    return float(torch.cat(fi_chunks).amin().item())


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


def make_prob(
    circuit: QuantumCircuit,
    period: int,
    *,
    exact_support: bool = False,
) -> ProbFunc:
    unitary = np.asarray(Operator(circuit).data)
    cache: dict[int, FloatArray] = {}

    def prob(col: int, shift: int) -> float:
        values = cache.setdefault(
            shift,
            (
                _probability_vector_from_support(
                    unitary,
                    _exact_support_indices(unitary.shape[1], period, shift),
                )
                if exact_support
                else _probability_vector(unitary, period, shift)
            ),
        )
        return float(values[col])

    return prob


def circuit_probability_distribution(
    circuit: QuantumCircuit,
    period: int,
    shift: int = 0,
    *,
    exact_support: bool = False,
) -> FloatArray:
    size = 1 << circuit.num_qubits
    return probability_distribution(
        make_prob(circuit, period, exact_support=exact_support),
        size,
        shift=shift,
    )


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
