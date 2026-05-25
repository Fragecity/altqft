from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

import altqft.circuits.HPgenerators as hp
from altqft.nn.process_qc import make_prob


def _is_shift_invariant(circuit: QuantumCircuit, *, column: int, period: int) -> bool:
    probability = make_prob(circuit, period)
    reference = probability(column, 0)
    return all(np.isclose(probability(column, shift), reference) for shift in range(1, period))


def _sample_HPcircuit(nqubit: int, *, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    max_layer = int(rng.integers(2, nqubit + 1))
    hlayout = rng.integers(0, max_layer, size=nqubit).tolist()
    phases = rng.uniform(0.0, 2.0 * np.pi, size=nqubit**2)
    return hp.HPqc(hlayout, phases)


@pytest.mark.parametrize(("nqubit", "seed"), [(3, 0), (3, 1), (4, 2), (5, 3)])
def test_HPcircuit_is_shift_invariant_for_power_of_two_periods(
    nqubit: int,
    seed: int,
) -> None:
    circuit = _sample_HPcircuit(nqubit, seed=seed)
    probe_rng = np.random.default_rng(seed + 100)
    state_space_size = 1 << nqubit

    for exponent in range(1, nqubit):
        period = 2**exponent
        column = int(probe_rng.integers(0, state_space_size))
        assert _is_shift_invariant(circuit, column=column, period=period)


@pytest.mark.parametrize("nqubit", [3, 4, 5])
def test_power_of_two_periods_divide_the_state_space(nqubit: int) -> None:
    state_space_size = 1 << nqubit
    for exponent in range(1, nqubit):
        assert state_space_size % (2**exponent) == 0
