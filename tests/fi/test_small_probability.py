import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from altqft.circuits.HPgenerators import HP1
from altqft.fi.small_probability import (
    dyadic_zero_fraction_lower_bound,
    hp1_log2_scaled_probabilities,
    normalization_fraction_lower_bound,
    soft_tail_dfi_samples,
    two_adic_parts,
)


def all_output_bits(nqubit: int) -> np.ndarray:
    values = np.arange(1 << nqubit, dtype=np.uint64)
    shifts = np.arange(nqubit, dtype=np.uint64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


@pytest.mark.parametrize("nqubit", [3, 4, 5, 6])
@pytest.mark.parametrize("period", [1, 2, 3, 4, 5, 6])
def test_root_filter_probabilities_match_hp1_statevector(
    nqubit: int,
    period: int,
) -> None:
    size = 1 << nqubit
    support_count = ((size - 1) // period) + 1
    state = np.zeros(size, dtype=np.complex128)
    state[0::period] = 1.0 / math.sqrt(support_count)
    expected = size * np.abs(Statevector(state).evolve(HP1(nqubit)).data) ** 2

    log2_scaled = hp1_log2_scaled_probabilities(
        all_output_bits(nqubit),
        period,
        output_chunk_size=7,
        residue_chunk_size=2,
    )
    actual = np.exp2(log2_scaled)

    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, (0, 1)), (2, (1, 1)), (12, (2, 3)), (40, (3, 5))],
)
def test_two_adic_parts(value: int, expected: tuple[int, int]) -> None:
    assert two_adic_parts(value) == expected


def test_normalization_fraction_lower_bound() -> None:
    assert normalization_fraction_lower_bound(math.log2(2.0)) == pytest.approx(0.5)
    assert normalization_fraction_lower_bound(math.log2(4.0)) == pytest.approx(0.75)
    assert normalization_fraction_lower_bound(0.0) == 0.0


def test_dyadic_zero_fraction_lower_bound() -> None:
    assert dyadic_zero_fraction_lower_bound(40, 2) == pytest.approx(1.0 - 2.0**-19)
    assert dyadic_zero_fraction_lower_bound(40, 4) == pytest.approx(1.0 - 2.0**-19)
    assert dyadic_zero_fraction_lower_bound(40, 12) == 0.0


@pytest.mark.parametrize("power", [1.0, 2.0, 8.0, math.inf])
def test_soft_tail_expectation_lower_bounds_exact_dfi(power: float) -> None:
    probability_p = np.array([0.05, 0.15, 0.30, 0.50])
    probability_q = np.array([0.10, 0.20, 0.25, 0.45])
    size = probability_p.size
    log2_scaled_p = np.log2(size * probability_p)
    log2_scaled_q = np.log2(size * probability_q)

    soft_bound = soft_tail_dfi_samples(
        log2_scaled_p,
        log2_scaled_q,
        threshold_c=2.0,
        power=power,
    ).mean()
    exact_dfi = np.sum(np.square(probability_q - probability_p) / probability_p)

    assert 0.0 < soft_bound <= exact_dfi
