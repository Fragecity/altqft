from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator

from altqft.algorithms.shor import (
    ShorConfig,
    _build_modular_exponentiation_circuit,
    _modular_multiplication_gate,
    _post_modular_exponentiation_state,
    _result_from_counts,
    _work_register_probabilities,
    run_shor,
)


def test_modular_multiplication_gate_maps_basis_states_for_15() -> None:
    operator = Operator(_modular_multiplication_gate(2, 15)).data

    for basis_state in range(15):
        target_state = (2 * basis_state) % 15
        assert operator[target_state, basis_state] == pytest.approx(1.0)
        assert operator[:, basis_state].sum() == pytest.approx(1.0)

    assert operator[15, 15] == pytest.approx(1.0)
    assert operator[:, 15].sum() == pytest.approx(1.0)


def test_result_from_counts_recovers_factors_from_valid_phases() -> None:
    config = ShorConfig(N=15, a=2, counting_qubits=8, shots=64, seed=7)
    counts = {
        "01000000": 9,
        "11000000": 7,
    }

    result = _result_from_counts(config, counts)

    assert result.success
    assert result.order == 4
    assert result.factors == (3, 5)
    assert result.candidates[0].bitstring == "01000000"
    assert result.candidates[0].validated_order == 4


def test_post_modexp_work_register_support_matches_ax_mod_n_values() -> None:
    config = ShorConfig(N=15, a=2, counting_qubits=8, shots=64, seed=7)

    state = _post_modular_exponentiation_state(config)
    probabilities = _work_register_probabilities(config, state)
    supported_values = {
        value
        for value, probability in enumerate(probabilities)
        if probability > 1e-12
    }

    assert supported_values == {1, 2, 4, 8}


def test_post_modexp_analytic_state_matches_circuit_work_distribution() -> None:
    config = ShorConfig(N=15, a=2, counting_qubits=4, shots=32, seed=7)

    analytic_state = _post_modular_exponentiation_state(config)
    circuit_state = Statevector.from_instruction(_build_modular_exponentiation_circuit(config))

    analytic_probabilities = _work_register_probabilities(config, analytic_state)
    circuit_probabilities = _work_register_probabilities(config, circuit_state)

    assert analytic_probabilities == pytest.approx(circuit_probabilities)


def test_run_shor_short_circuits_when_gcd_is_nontrivial() -> None:
    config = ShorConfig(N=15, a=5, counting_qubits=8, shots=32, seed=7)

    result = run_shor(config)

    assert result.success
    assert result.factors == (3, 5)
    assert result.order is None
    assert result.candidates == []


def test_run_shor_factors_15_end_to_end() -> None:
    config = ShorConfig(N=15, a=2, counting_qubits=8, shots=256, seed=7)

    result = run_shor(config)

    assert result.success
    assert result.factors == (3, 5)
    assert result.order == 4
    assert result.candidates


def test_cli_smoke_prints_candidates_and_final_factorization(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path("scripts/algorithms/run_shor.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--N",
            "15",
            "--a",
            "2",
            "--counting-qubits",
            "8",
            "--shots",
            "256",
            "--seed",
            "7",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "candidate bitstring=" in stdout
    assert "15 = 3 x 5" in stdout
