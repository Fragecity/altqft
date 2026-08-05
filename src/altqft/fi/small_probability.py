from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

BitArray = NDArray[np.uint8]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def two_adic_parts(value: int) -> tuple[int, int]:
    """Return ``(a, u)`` such that ``value = 2**a * u`` and ``u`` is odd."""
    if value < 1:
        raise ValueError("value must be positive")
    power = (value & -value).bit_length() - 1
    return power, value >> power


def uniform_output_bits(
    nqubit: int,
    sample_count: int,
    rng: np.random.Generator,
) -> BitArray:
    """Draw uniformly distributed output strings in Qiskit's qubit order."""
    return rng.integers(0, 2, size=(sample_count, nqubit), dtype=np.uint8)


def _hp1_input_digit_weights(output_bits: BitArray, first_digit: int) -> ComplexArray:
    nqubit = output_bits.shape[1]
    active_qubits = np.arange(first_digit, nqubit, dtype=np.int64)
    signs = 1.0 - 2.0 * output_bits[:, active_qubits].astype(np.float64)
    weights = signs.astype(np.complex128)

    even_qubits = np.arange(0, nqubit, 2, dtype=np.int64)
    odd_positions = np.flatnonzero(active_qubits % 2 == 1)
    if odd_positions.size == 0:
        return weights

    odd_qubits = active_qubits[odd_positions]
    distances = np.abs(even_qubits[:, None] - odd_qubits[None, :])
    phase_matrix = math.pi / np.exp2(distances.astype(np.float64))
    controlled_phase = (
        output_bits[:, even_qubits].astype(np.float64) @ phase_matrix
    )
    weights[:, odd_positions] *= np.exp(1j * controlled_phase)
    return weights


def _power_two_residues(width: int, modulus: int) -> NDArray[np.int64]:
    if modulus == 1:
        return np.zeros(width, dtype=np.int64)
    return np.fromiter(
        (pow(2, digit, modulus) for digit in range(width)),
        dtype=np.int64,
        count=width,
    )


def hp1_log2_scaled_probabilities(
    output_bits: BitArray,
    period: int,
    *,
    output_chunk_size: int = 256,
    residue_chunk_size: int = 256,
) -> FloatArray:
    """Evaluate ``log2(2**n * Pr_r(x))`` without a state vector.

    ``output_bits[:, j]`` is the bit on Qiskit qubit ``j`` (little-endian
    integer significance).  The calculation is exact up to complex128
    arithmetic and costs ``O(M * (n-a) * u)``, where ``M`` is the number of
    rows and ``period = 2**a * u`` with odd ``u``.
    """
    if output_bits.ndim != 2 or output_bits.shape[1] < 1:
        raise ValueError("output_bits must have shape (sample_count, nqubit)")
    if output_chunk_size < 1 or residue_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")

    nqubit = output_bits.shape[1]
    size = 1 << nqubit
    if period > size:
        raise ValueError("period must not exceed 2**nqubit")

    two_power, odd_part = two_adic_parts(period)
    digit_weights = _hp1_input_digit_weights(output_bits, two_power)
    width = nqubit - two_power
    residues = _power_two_residues(width, odd_part).astype(np.float64)
    period_sums = np.zeros(output_bits.shape[0], dtype=np.complex128)

    for residue_start in range(0, odd_part, residue_chunk_size):
        residue_stop = min(residue_start + residue_chunk_size, odd_part)
        frequencies = np.arange(
            residue_start,
            residue_stop,
            dtype=np.float64,
        )
        roots = np.exp(
            (2j * math.pi / float(odd_part))
            * frequencies[:, None]
            * residues[None, :]
        )

        for output_start in range(0, output_bits.shape[0], output_chunk_size):
            output_stop = min(
                output_start + output_chunk_size,
                output_bits.shape[0],
            )
            weights = digit_weights[output_start:output_stop]
            products = np.ones(
                (weights.shape[0], frequencies.size),
                dtype=np.complex128,
            )
            for digit in range(width):
                products *= 0.5 * (
                    1.0 + weights[:, digit, None] * roots[None, :, digit]
                )
            period_sums[output_start:output_stop] += products.sum(axis=1)

    normalized_sum = np.abs(period_sums / float(odd_part))
    support_count = ((size - 1) // period) + 1
    with np.errstate(divide="ignore"):
        return np.asarray(
            2.0 * float(nqubit - two_power)
            - math.log2(support_count)
            + 2.0 * np.log2(normalized_sum),
            dtype=np.float64,
        )


def small_probability_fraction(
    log2_scaled_probabilities: FloatArray,
    log2_scaled_threshold: float,
) -> float:
    """Return the fraction satisfying ``Pr(x) <= 2**threshold / 2**n``."""
    return float(np.mean(log2_scaled_probabilities <= log2_scaled_threshold))


def soft_tail_dfi_samples(
    log2_scaled_p: FloatArray,
    log2_scaled_q: FloatArray,
    *,
    threshold_c: float,
    power: float = 1.0,
) -> FloatArray:
    """Pointwise samples of a soft-tail lower bound on the exact DFI.

    For uniform output strings and ``p = 2**n P_r``, ``q = 2**n P_{r+1}``, the
    returned random variable is

    ``(q-p)**2 / C * max(0, 1 - (p/C)**power)``.

    Its uniform expectation is no larger than the exact DFI.  Setting
    ``power=math.inf`` gives the hard cutoff ``(q-p)**2 / C * 1{p<C}``.
    """
    if threshold_c <= 0.0 or power <= 0.0:
        raise ValueError("threshold_c and power must be positive")
    if log2_scaled_p.shape != log2_scaled_q.shape:
        raise ValueError("p and q arrays must have the same shape")

    log2_c = math.log2(threshold_c)
    active = log2_scaled_p < log2_c
    samples = np.zeros_like(log2_scaled_p, dtype=np.float64)
    scaled_p = np.exp2(log2_scaled_p[active])
    scaled_q = np.exp2(log2_scaled_q[active])
    margin = (
        np.ones_like(scaled_p)
        if math.isinf(power)
        else 1.0 - np.power(scaled_p / threshold_c, power)
    )
    samples[active] = (
        np.square(scaled_q - scaled_p) * margin / threshold_c
    )
    return samples


def hoeffding_radius(sample_count: int, confidence: float) -> float:
    """Two-sided additive radius for a Bernoulli mean."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    return math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * sample_count))


def normalization_fraction_lower_bound(log2_scaled_threshold: float) -> float:
    """Universal lower bound from ``sum_x Pr(x) = 1``."""
    if log2_scaled_threshold <= 0.0:
        return 0.0
    return max(0.0, 1.0 - math.exp2(-log2_scaled_threshold))


def dyadic_zero_fraction_lower_bound(nqubit: int, period: int) -> float:
    """Fraction forced to have zero probability when ``period`` is dyadic."""
    if nqubit < 1 or period > 1 << nqubit:
        raise ValueError("expected nqubit >= 1 and period <= 2**nqubit")
    two_power, odd_part = two_adic_parts(period)
    if odd_part != 1:
        return 0.0
    constrained_even_bits = (nqubit + 1) // 2 - (two_power + 1) // 2
    return 1.0 - math.exp2(-constrained_even_bits)
