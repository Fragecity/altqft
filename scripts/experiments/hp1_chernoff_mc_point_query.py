#!/usr/bin/env python3
"""Monte Carlo C_1/2 estimates using pointwise HP-1 probabilities.

Unlike ``hp1_chernoff_mc_compare.py``, this script does not build full
probability vectors.  It samples output strings x and evaluates P_r(x),
P_{r+1}(x) from the Appendix E amplitude formula:

    P_r(x) = |sum_q exp(i phi(x, q r))|^2 / (2^n R_r).

For a proposal distribution Q on output strings, the estimator is

    B_1/2 = E_{x ~ Q} sqrt(P_r(x) P_{r+1}(x)) / Q(x),
    C_1/2 = -log B_1/2.

The default proposal is a defensive mixture: uniform samples plus samples
from zero-prefix subspaces.  It keeps the estimator unbiased while reducing
the variance of dyadic spikes such as r=2^k.  This removes the explicit 2^n
output loop, but each sampled x still costs O(2^n / r) phase terms for period
r unless the support count is capped by the caller.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


UIntArray = NDArray[np.uint64]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ExactRow:
    r: int
    chernoff_half: float


@dataclass(frozen=True)
class McRow:
    n: int
    r: int
    proposal: str
    support_count_r: int
    support_count_next: int
    sample_count: int
    seed: int
    exact_chernoff_half: float
    mc_chernoff_half: float
    mc_standard_error: float
    abs_error: float
    rel_error: float
    coefficient_estimate: float
    coefficient_standard_error: float
    seconds: float


def bit_matrix(values: UIntArray, nqubit: int) -> NDArray[np.uint8]:
    if nqubit > 64:
        raise ValueError("uint64 bit_matrix only supports n <= 64")
    shifts = np.arange(nqubit - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def bit_matrix_from_ints(values: Sequence[int], nqubit: int) -> BitArray:
    bits = np.empty((len(values), nqubit), dtype=np.uint8)
    for row, value in enumerate(values):
        bits[row] = [(value >> shift) & 1 for shift in range(nqubit - 1, -1, -1)]
    return bits


class HP1PointProbability:
    def __init__(
        self,
        nqubit: int,
        period: int,
        *,
        max_support_count: int | None = None,
    ) -> None:
        self.nqubit = nqubit
        self.period = period
        self.size = 1 << nqubit
        self.support_count = ((self.size - 1) // period) + 1
        if max_support_count is not None and self.support_count > max_support_count:
            raise ValueError(
                f"period {period} has support_count={self.support_count}, "
                f"above max_support_count={max_support_count}"
            )
        self.support_bits = self._build_support_bits()
        self.odd_support_bits = self.support_bits[:, 1::2].astype(np.float64)
        self.support_bits_t = self.support_bits.T.astype(np.float64)
        self.phase_weights = self._phase_weights()

    def _build_support_bits(self) -> BitArray:
        max_uint64 = np.iinfo(np.uint64).max
        largest_support = (self.support_count - 1) * self.period
        if self.nqubit <= 64 and largest_support <= max_uint64:
            support = (
                np.arange(self.support_count, dtype=np.uint64) * np.uint64(self.period)
            )
            return bit_matrix(support, self.nqubit)
        return bit_matrix_from_ints(
            [q_value * self.period for q_value in range(self.support_count)],
            self.nqubit,
        )

    def _phase_weights(self) -> FloatArray:
        weights = np.zeros((self.nqubit, self.support_count), dtype=np.float64)
        weights += math.pi * self.support_bits_t
        for control in range(0, self.nqubit, 2):
            for odd_index, target in enumerate(range(1, self.nqubit, 2)):
                weights[control] += (
                    math.pi / (2 ** abs(target - control))
                ) * self.odd_support_bits[:, odd_index]
        return weights

    def probability(self, x_values: UIntArray, *, chunk_size: int) -> FloatArray:
        return self.probability_bits(
            bit_matrix(x_values, self.nqubit),
            chunk_size=chunk_size,
        )

    def probability_bits(self, x_bits: BitArray, *, chunk_size: int) -> FloatArray:
        probabilities = np.empty(x_bits.shape[0], dtype=np.float64)
        for start in range(0, x_bits.shape[0], chunk_size):
            stop = min(start + chunk_size, x_bits.shape[0])
            chunk_bits = x_bits[start:stop].astype(np.float64, copy=False)
            phases = chunk_bits @ self.phase_weights
            amplitudes = np.exp(1j * phases).sum(axis=1)
            probabilities[start:stop] = (
                np.square(np.abs(amplitudes))
                / (float(self.size) * float(self.support_count))
            )
        return probabilities


def load_exact_rows(path: Path) -> dict[int, ExactRow]:
    with path.open(newline="") as handle:
        return {
            int(row["r"]): ExactRow(
                r=int(row["r"]),
                chernoff_half=float(row["chernoff_half"]),
            )
            for row in csv.DictReader(handle)
        }


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def default_prefixes(nqubit: int) -> tuple[int, ...]:
    step = max(4, nqubit // 8)
    prefixes = {value for value in (4, 8, 12, 16) if value <= nqubit}
    prefixes.update(range(step, nqubit + 1, step))
    prefixes.add(nqubit)
    return tuple(sorted(prefixes))


def parse_prefixes(raw_value: str, nqubit: int) -> tuple[int, ...]:
    if raw_value == "auto":
        return default_prefixes(nqubit)
    if raw_value in {"", "none"}:
        return ()
    values = tuple(sorted(set(parse_int_list(raw_value))))
    for value in values:
        if value < 1 or value > nqubit:
            raise ValueError(f"prefix {value} is outside 1..{nqubit}")
    return values


def leading_zero_counts(x_bits: BitArray) -> NDArray[np.int64]:
    nonzero = x_bits != 0
    has_one = nonzero.any(axis=1)
    first_one = np.argmax(nonzero, axis=1).astype(np.int64)
    counts = np.where(has_one, first_one, x_bits.shape[1])
    return np.asarray(counts, dtype=np.int64)


def proposal_description(prefixes: Sequence[int], uniform_weight: float) -> str:
    if not prefixes:
        return "uniform"
    prefix_text = ",".join(str(value) for value in prefixes)
    return f"prefix_mix(uniform={uniform_weight:.3g};prefixes={prefix_text})"


def sample_prefix_mixture(
    nqubit: int,
    *,
    sample_count: int,
    rng: np.random.Generator,
    prefixes: Sequence[int],
    uniform_weight: float,
) -> tuple[BitArray, FloatArray, str]:
    if not prefixes:
        x_bits = rng.integers(0, 2, size=(sample_count, nqubit), dtype=np.uint8)
        return x_bits, np.full(sample_count, 2.0 ** (-nqubit), dtype=np.float64), "uniform"

    if not 0.0 < uniform_weight < 1.0:
        raise ValueError("uniform_weight must be in (0, 1) when prefixes are used")

    components = [-1, *prefixes]
    prefix_weight = (1.0 - uniform_weight) / len(prefixes)
    weights = np.array(
        [uniform_weight, *([prefix_weight] * len(prefixes))],
        dtype=np.float64,
    )
    component_indices = rng.choice(len(components), size=sample_count, p=weights)
    x_bits = np.zeros((sample_count, nqubit), dtype=np.uint8)

    for component_index, component in enumerate(components):
        row_indices = np.flatnonzero(component_indices == component_index)
        if row_indices.size == 0:
            continue
        if component == -1:
            x_bits[row_indices] = rng.integers(
                0,
                2,
                size=(row_indices.size, nqubit),
                dtype=np.uint8,
            )
            continue
        suffix_width = nqubit - component
        if suffix_width > 0:
            x_bits[row_indices, component:] = rng.integers(
                0,
                2,
                size=(row_indices.size, suffix_width),
                dtype=np.uint8,
            )

    lz_counts = leading_zero_counts(x_bits)
    proposal_probability = np.full(
        sample_count,
        uniform_weight * (2.0 ** (-nqubit)),
        dtype=np.float64,
    )
    for prefix in prefixes:
        proposal_probability[lz_counts >= prefix] += prefix_weight * (
            2.0 ** (-(nqubit - prefix))
        )

    return (
        x_bits,
        proposal_probability,
        proposal_description(prefixes, uniform_weight),
    )


def importance_estimate(
    nqubit: int,
    period: int,
    *,
    exact_chernoff_half: float = math.nan,
    sample_count: int,
    seed: int,
    chunk_size: int,
    prefixes: Sequence[int] = (),
    uniform_weight: float = 1.0,
    max_support_count: int | None = None,
) -> McRow:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    x_bits, proposal_probability, proposal = sample_prefix_mixture(
        nqubit,
        sample_count=sample_count,
        rng=rng,
        prefixes=prefixes,
        uniform_weight=uniform_weight,
    )
    point_probability_p = HP1PointProbability(
        nqubit,
        period,
        max_support_count=max_support_count,
    )
    point_probability_q = HP1PointProbability(
        nqubit,
        period + 1,
        max_support_count=max_support_count,
    )
    probability_p = point_probability_p.probability_bits(
        x_bits,
        chunk_size=chunk_size,
    )
    probability_q = point_probability_q.probability_bits(
        x_bits,
        chunk_size=chunk_size,
    )
    samples = np.sqrt(probability_p * probability_q) / proposal_probability
    coefficient = float(samples.mean())
    coefficient_se = float(samples.std(ddof=1) / math.sqrt(sample_count))
    chernoff_half = -math.log(max(coefficient, np.finfo(np.float64).tiny))
    chernoff_se = coefficient_se / max(coefficient, np.finfo(np.float64).tiny)
    if math.isfinite(exact_chernoff_half):
        abs_error = abs(chernoff_half - exact_chernoff_half)
        rel_error = abs_error / exact_chernoff_half
    else:
        abs_error = math.nan
        rel_error = math.nan
    return McRow(
        n=nqubit,
        r=period,
        proposal=proposal,
        support_count_r=point_probability_p.support_count,
        support_count_next=point_probability_q.support_count,
        sample_count=sample_count,
        seed=seed,
        exact_chernoff_half=exact_chernoff_half,
        mc_chernoff_half=chernoff_half,
        mc_standard_error=chernoff_se,
        abs_error=abs_error,
        rel_error=rel_error,
        coefficient_estimate=coefficient,
        coefficient_standard_error=coefficient_se,
        seconds=time.perf_counter() - started,
    )


def mc_estimate(
    nqubit: int,
    period: int,
    *,
    exact_chernoff_half: float,
    sample_count: int,
    seed: int,
    chunk_size: int,
) -> McRow:
    return importance_estimate(
        nqubit,
        period,
        exact_chernoff_half=exact_chernoff_half,
        sample_count=sample_count,
        seed=seed,
        chunk_size=chunk_size,
        prefixes=(),
        uniform_weight=1.0,
    )


def write_csv(path: Path, rows: list[McRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(McRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument(
        "--periods",
        default="53,106,142,165,221,244,256",
        help="Comma-separated r values to compare against exact CSV.",
    )
    parser.add_argument("--sample-counts", default="1000,5000,20000")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--proposal",
        choices=("uniform", "prefix"),
        default="prefix",
        help="Sampling proposal for output strings x.",
    )
    parser.add_argument(
        "--prefixes",
        default="auto",
        help="Comma-separated zero-prefix depths, 'auto', or 'none'.",
    )
    parser.add_argument(
        "--uniform-weight",
        type=float,
        default=0.5,
        help="Defensive uniform mixture weight for --proposal prefix.",
    )
    parser.add_argument(
        "--max-support-count",
        type=int,
        default=0,
        help="Abort if R_r exceeds this value; 0 means no cap.",
    )
    parser.add_argument(
        "--exact-csv",
        type=Path,
        default=Path("data/hp1_chernoff/repro_sampled_cuda/hp1_appendix_n16_logspace.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hp1_chernoff/mc_point_query/hp1_n16_uniform_mc.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_rows = load_exact_rows(args.exact_csv)
    prefixes = (
        parse_prefixes(args.prefixes, args.n) if args.proposal == "prefix" else ()
    )
    uniform_weight = args.uniform_weight if prefixes else 1.0
    max_support_count = args.max_support_count or None
    rows: list[McRow] = []
    for period in parse_int_list(args.periods):
        if period not in exact_rows:
            raise ValueError(f"period {period} is not present in {args.exact_csv}")
        for sample_count in parse_int_list(args.sample_counts):
            row = importance_estimate(
                args.n,
                period,
                exact_chernoff_half=exact_rows[period].chernoff_half,
                sample_count=sample_count,
                seed=args.seed + period * 1000003 + sample_count,
                chunk_size=args.chunk_size,
                prefixes=prefixes,
                uniform_weight=uniform_weight,
                max_support_count=max_support_count,
            )
            rows.append(row)
            print(
                "r={r:<4d} M={sample_count:<7d} exact={exact_chernoff_half:.6f} "
                "mc={mc_chernoff_half:.6f} +/- {mc_standard_error:.4f} "
                "abs={abs_error:.4f} rel={rel_error:.2%} seconds={seconds:.2f}".format(
                    **row.__dict__
                ),
                flush=True,
            )
    write_csv(args.output, rows)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
