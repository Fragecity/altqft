from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from altqft.algorithms.shor_ph1 import (
    PH1ShorConfig,
    _collapsed_periodic_state,
    _modular_exponentiation_outputs,
)
from altqft.circuits.ph_generators import qft
from altqft.nn.process_qc import (
    _exact_support_indices,
    _surrogate_support_indices,
    _torch_exact_support_indices,
    _torch_probability_vector_from_support,
    _torch_unitary,
    circuit_probability_distribution,
    fi,
    min_fi,
)


def test_fi_returns_zero_for_identical_distributions() -> None:
    assert math.isclose(fi([0.5, 0.5], [0.5, 0.5]), 0.0, abs_tol=1e-9)


def test_fi_matches_hand_computed_value() -> None:
    actual = fi([0.8, 0.2], [0.6, 0.4])
    assert math.isclose(actual, 0.25, rel_tol=1e-9)


def _mock_make_prob(circuit: QuantumCircuit, period: int) -> Callable[[int, int], float]:
    def prob(col: int, shift: int) -> float:
        del shift
        if period == 2:
            return 0.8 if col == 0 else 0.2
        if period == 3:
            return 0.6 if col == 0 else 0.4
        if period == 4:
            return 0.5 if col == 0 else 0.5
        return 0.5

    return prob


def test_min_fi_tracks_the_worst_adjacent_period_pair() -> None:
    with patch("altqft.nn.process_qc.make_prob", side_effect=_mock_make_prob):
        actual = min_fi(QuantumCircuit(1), period_range=[2, 3])
    assert math.isclose(actual, 1.0 / 24.0, rel_tol=1e-9)


def test_min_fi_torch_cpu_matches_numpy() -> None:
    circuit = qft(3)
    period_range = [3, 4, 5]

    expected = min_fi(circuit, period_range)
    actual = min_fi(circuit, period_range, device="cpu")

    assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)


def _statevector_distribution_from_support(
    circuit: QuantumCircuit,
    support: np.ndarray,
) -> np.ndarray:
    amplitudes = np.zeros(1 << circuit.num_qubits, dtype=np.complex128)
    amplitudes[support] = 1.0 / math.sqrt(len(support))
    return np.asarray(Statevector(amplitudes).evolve(circuit).probabilities(), dtype=np.float64)


def test_circuit_probability_distribution_matches_surrogate_statevector_evolution() -> None:
    circuit = qft(3)
    period = 3
    shift = 1
    size = 1 << circuit.num_qubits
    support = _surrogate_support_indices(size, period, shift)
    expected = _statevector_distribution_from_support(circuit, support)

    actual = circuit_probability_distribution(circuit, period, shift=shift)

    assert np.allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize(("num_qubits", "period"), [(3, 3), (4, 5), (4, 6)])
def test_exact_support_mismatch_count_matches_state_space_remainder(
    num_qubits: int,
    period: int,
) -> None:
    size = 1 << num_qubits
    mismatched_shifts = [
        shift
        for shift in range(period)
        if not np.array_equal(
            _surrogate_support_indices(size, period, shift),
            _exact_support_indices(size, period, shift),
        )
    ]

    assert mismatched_shifts == list(range(size % period))


def test_surrogate_and_exact_supports_diverge_for_non_power_of_two_period() -> None:
    circuit = qft(3)
    period = 3
    size = 1 << circuit.num_qubits
    mismatched_shifts: list[int] = []

    for shift in range(period):
        surrogate_support = _surrogate_support_indices(size, period, shift)
        exact_support = _exact_support_indices(size, period, shift)
        surrogate = circuit_probability_distribution(circuit, period, shift=shift)
        exact = _statevector_distribution_from_support(circuit, exact_support)
        if not np.array_equal(surrogate_support, exact_support):
            mismatched_shifts.append(shift)
            assert not np.allclose(surrogate, exact, atol=1e-9)
        else:
            assert np.allclose(surrogate, exact, atol=1e-9)

    assert mismatched_shifts == [0, 1]


def test_surrogate_and_exact_supports_match_for_power_of_two_period() -> None:
    circuit = qft(3)
    period = 4
    size = 1 << circuit.num_qubits

    for shift in range(period):
        surrogate_support = _surrogate_support_indices(size, period, shift)
        exact_support = _exact_support_indices(size, period, shift)
        surrogate = circuit_probability_distribution(circuit, period, shift=shift)
        exact = _statevector_distribution_from_support(circuit, exact_support)

        assert np.array_equal(surrogate_support, exact_support)
        assert np.allclose(surrogate, exact, atol=1e-9)


@pytest.mark.parametrize(
    ("nqubit", "N", "a", "true_period"),
    [(5, 21, 2, 6), (6, 35, 2, 12)],
)
def test_exact_support_helper_matches_collapsed_shor_statevector(
    tmp_path: Path,
    nqubit: int,
    N: int,
    a: int,
    true_period: int,
) -> None:
    config = PH1ShorConfig(
        N=N,
        a=a,
        nqubit=nqubit,
        period_min=nqubit,
        period_max=true_period,
        measurement_count=1,
        top_k=1,
        seed=7,
        model_dir=tmp_path / "model",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        allow_phase_retraining=False,
    )
    outputs = _modular_exponentiation_outputs(config)
    size = 1 << nqubit
    remainder = size % true_period
    work_value = next(
        int(value)
        for value in np.unique(outputs)
        if int(np.flatnonzero(outputs == value)[0]) < remainder
    )
    collapsed_state, support = _collapsed_periodic_state(
        config,
        outputs=outputs,
        work_value=work_value,
    )
    shift = min(support)
    circuit = qft(nqubit)
    unitary = _torch_unitary(circuit, "cpu")

    exact_support = _exact_support_indices(size, true_period, shift)
    surrogate_support = _surrogate_support_indices(size, true_period, shift)
    helper_exact_prob = _torch_probability_vector_from_support(
        unitary,
        _torch_exact_support_indices(size, true_period, shift, device=unitary.device),
    ).detach().cpu().numpy()
    exact_prob = np.asarray(collapsed_state.evolve(circuit).probabilities(), dtype=np.float64)
    surrogate_prob = _statevector_distribution_from_support(circuit, surrogate_support)

    assert remainder > 0
    assert tuple(exact_support.tolist()) == support
    assert surrogate_support.shape[0] < exact_support.shape[0]
    assert np.allclose(helper_exact_prob, exact_prob, atol=1e-9)
    assert not np.allclose(surrogate_prob, exact_prob, atol=1e-9)