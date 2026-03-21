import numpy as np
import pytest
from qiskit.quantum_info import Operator

from altqft.circuits.ph_generators import ph_1_hlayout, ph_1_parametrized


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
