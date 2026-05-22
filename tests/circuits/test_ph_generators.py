import numpy as np
import pytest
from qiskit.quantum_info import Operator

from altqft.circuits.ph_generators import (
    HP1_shared_parameter,
    ph_1_hlayout,
    ph_1_parametrized,
)


def test_ph_1_hlayout_pattern() -> None:
    assert ph_1_hlayout(5) == [0, 1, 0, 1, 0]


def test_ph_1_parametrized_builds_expected_circuit() -> None:
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    circuit = ph_1_parametrized(4, phases)
    operator = Operator(circuit).data
    assert operator.shape == (16, 16)


def test_ph_1_parametrized_rejects_bad_phase_count() -> None:
    with pytest.raises(ValueError):
        ph_1_parametrized(4, [0.1, 0.2])


def test_HP1_shared_parameter_uses_nearby_shared_phases() -> None:
    phases = np.array([0.1, 0.2])
    circuit = HP1_shared_parameter(6, phases)

    cp_ops = []
    for instruction in circuit.data:
        if instruction.operation.name == "cp":
            control = circuit.find_bit(instruction.qubits[0]).index
            target = circuit.find_bit(instruction.qubits[1]).index
            phase = float(instruction.operation.params[0])
            cp_ops.append((control, target, phase))

    assert cp_ops == [
        (0, 1, 0.2),
        (0, 5, 0.1),
        (2, 1, 0.2),
        (2, 3, 0.2),
        (4, 3, 0.2),
        (4, 5, 0.2),
    ]


def test_HP1_shared_parameter_rejects_bad_phase_count() -> None:
    with pytest.raises(ValueError):
        HP1_shared_parameter(6, [0.1])
