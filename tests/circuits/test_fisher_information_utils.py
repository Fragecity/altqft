from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

ROOT_DIR = Path(__file__).resolve().parents[2]
FI_SCRIPT_DIR = ROOT_DIR / "scripts" / "fi_data_cal"
if str(FI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FI_SCRIPT_DIR))

from fisher_information_utils import build_circuit


@pytest.mark.parametrize(
    ("circuit_type", "nlayer"),
    [
        ("qft", None),
        ("ph1", None),
        ("ph_random", 3),
        ("ph_1_random", None),
        ("ph_random_phase", 3),
    ],
)
def test_build_circuit_supports_all_generators(
    circuit_type: str,
    nlayer: int | None,
) -> None:
    circuit = build_circuit(circuit_type, nqubit=4, nlayer=nlayer)

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4


@pytest.mark.parametrize("circuit_type", ["ph_random", "ph_random_phase"])
def test_build_circuit_requires_nlayer_for_random_layouts(circuit_type: str) -> None:
    with pytest.raises(ValueError, match="nlayer"):
        build_circuit(circuit_type, nqubit=4)
