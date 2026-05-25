from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from altqft.fi.dataset import build_circuit


@pytest.mark.parametrize(
    "circuit_type",
    [
        "qft",
        "ph1",
        "HP1_random",
    ],
)
def test_build_circuit_supports_all_generators(
    circuit_type: str,
) -> None:
    circuit = build_circuit(circuit_type, nqubit=4)

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4
