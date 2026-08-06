#!/usr/bin/env python3
"""Exact finite-window counts for the HP-1 active small-denominator set.

For ``N=2**n``, ``p=N P_r`` and ``q=N P_{r+1}``, count

    A_tau(n,r) = {x: p(x)<2 and (q(x)-p(x))**2 r**2 >= tau N}.

Every counted point contributes at least ``tau/(2 r**2)`` to exact DFI.
This script enumerates all outputs and is only a finite-size validator for the
rare-event CUDA estimator in ``hp1_active_tail_subset_cuda.cu``.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from altqft.fi.small_probability import hp1_log2_scaled_probabilities, two_adic_parts


@dataclass(frozen=True)
class ActiveTailRow:
    n: int
    period: int
    odd_part: int
    tau: float
    active_count: int
    active_fraction: float
    normalized_fraction: float
    log_normalized_fraction: float
    log_dfi_count_bound: float


def output_bits(start: int, stop: int, nqubit: int) -> np.ndarray:
    values = np.arange(start, stop, dtype=np.uint64)
    shifts = np.arange(nqubit, dtype=np.uint64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def period_limit(nqubit: int) -> int:
    """Largest integer satisfying r < 2**(n/4)."""
    return math.ceil(math.exp2(nqubit / 4.0)) - 1


def scan_nqubit(
    nqubit: int,
    *,
    tau: float,
    chunk_size: int,
    even_non_dyadic_only: bool,
) -> list[ActiveTailRow]:
    size = 1 << nqubit
    maximum_period = period_limit(nqubit)
    periods = [
        period
        for period in range(2, maximum_period + 1)
        if not even_non_dyadic_only
        or (period % 2 == 0 and two_adic_parts(period)[1] > 1)
    ]
    counts = {period: 0 for period in periods}
    needed_periods = sorted({value for period in periods for value in (period, period + 1)})

    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        bits = output_bits(start, stop, nqubit)
        log_probabilities = {
            period: hp1_log2_scaled_probabilities(
                bits,
                period,
                output_chunk_size=1024,
            )
            for period in needed_periods
        }
        for period in periods:
            scaled_p = np.exp2(log_probabilities[period])
            scaled_q = np.exp2(log_probabilities[period + 1])
            scaled_difference_squared = np.square(scaled_q - scaled_p)
            active = (scaled_p < 2.0) & (
                scaled_difference_squared * float(period * period) >= tau * float(size)
            )
            counts[period] += int(np.count_nonzero(active))

    rows: list[ActiveTailRow] = []
    for period in periods:
        fraction = counts[period] / float(size)
        normalized = fraction / float(period * period)
        log_normalized = math.log(normalized) if normalized > 0.0 else -math.inf
        rows.append(
            ActiveTailRow(
                n=nqubit,
                period=period,
                odd_part=two_adic_parts(period)[1],
                tau=tau,
                active_count=counts[period],
                active_fraction=fraction,
                normalized_fraction=normalized,
                log_normalized_fraction=log_normalized,
                log_dfi_count_bound=(
                    math.log(tau / 2.0) + nqubit * math.log(2.0) + log_normalized
                ),
            )
        )
    return rows


def write_rows(path: Path, rows: list[ActiveTailRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--tau", type=float, default=3.0e-4)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--all-periods", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hp1_active_tail_exact/even_non_dyadic_n10_20.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[ActiveTailRow] = []
    for nqubit in range(args.n_min, args.n_max + 1):
        current = scan_nqubit(
            nqubit,
            tau=args.tau,
            chunk_size=args.chunk_size,
            even_non_dyadic_only=not args.all_periods,
        )
        rows.extend(current)
        finite = [row for row in current if math.isfinite(row.log_dfi_count_bound)]
        minimum = min(finite, key=lambda row: row.log_dfi_count_bound) if finite else None
        if minimum is None:
            print(f"n={nqubit}: no eligible periods")
        else:
            print(
                f"n={nqubit} r_min={minimum.period} "
                f"active={minimum.active_fraction:.9g} "
                f"log_DFI_lb={minimum.log_dfi_count_bound:.9g}",
                flush=True,
            )
    if not rows:
        raise ValueError("the selected range contains no eligible periods")
    write_rows(args.output, rows)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
