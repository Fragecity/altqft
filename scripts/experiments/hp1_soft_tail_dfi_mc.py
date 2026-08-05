#!/usr/bin/env python3
"""Statevector-free Monte Carlo lower bound for fixed-phase HP-1 DFI.

For uniform output strings X, set p(X) = 2**n P_r(X) and
q(X) = 2**n P_{r+1}(X).  This script estimates

    F_{s,C} = E[(q-p)^2/C * (1-(p/C)^s)_+],

which is a pointwise lower bound on the exact adjacent-period DFI.  HP-1 point
probabilities use the roots-of-unity filter, so no 2**n state vector is built.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from altqft.fi.small_probability import (
    hp1_log2_scaled_probabilities,
    soft_tail_dfi_samples,
    two_adic_parts,
    uniform_output_bits,
)


@dataclass(frozen=True)
class SoftTailRow:
    n: int
    period: int
    period_odd_part: int
    next_period_odd_part: int
    threshold_c: float
    power: float
    sample_count: int
    estimate: float
    standard_error: float
    effective_sample_size: float
    maximum_weight_fraction: float
    active_fraction: float
    seconds: float


def parse_integer_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_float_list(raw_value: str) -> list[float]:
    values = [float(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one number")
    return values


def summarize_samples(
    nqubit: int,
    period: int,
    threshold_c: float,
    power: float,
    samples: np.ndarray,
    *,
    active_fraction: float,
    seconds: float,
) -> SoftTailRow:
    sample_count = samples.size
    total = float(samples.sum())
    squared_total = float(np.dot(samples, samples))
    effective_sample_size = total * total / squared_total if squared_total > 0.0 else 0.0
    maximum_weight_fraction = float(samples.max()) / total if total > 0.0 else 0.0
    return SoftTailRow(
        n=nqubit,
        period=period,
        period_odd_part=two_adic_parts(period)[1],
        next_period_odd_part=two_adic_parts(period + 1)[1],
        threshold_c=threshold_c,
        power=power,
        sample_count=sample_count,
        estimate=float(samples.mean()),
        standard_error=float(samples.std(ddof=1) / math.sqrt(sample_count)),
        effective_sample_size=effective_sample_size,
        maximum_weight_fraction=maximum_weight_fraction,
        active_fraction=active_fraction,
        seconds=seconds,
    )


def estimate_period(
    output_bits: np.ndarray,
    period: int,
    *,
    threshold_c: float,
    powers: list[float],
    output_chunk_size: int,
    residue_chunk_size: int,
) -> list[SoftTailRow]:
    started = time.perf_counter()
    log2_p = hp1_log2_scaled_probabilities(
        output_bits,
        period,
        output_chunk_size=output_chunk_size,
        residue_chunk_size=residue_chunk_size,
    )
    log2_q = hp1_log2_scaled_probabilities(
        output_bits,
        period + 1,
        output_chunk_size=output_chunk_size,
        residue_chunk_size=residue_chunk_size,
    )
    elapsed = time.perf_counter() - started
    active_fraction = float(np.mean(log2_p < math.log2(threshold_c)))
    return [
        summarize_samples(
            output_bits.shape[1],
            period,
            threshold_c,
            power,
            soft_tail_dfi_samples(
                log2_p,
                log2_q,
                threshold_c=threshold_c,
                power=power,
            ),
            active_fraction=active_fraction,
            seconds=elapsed,
        )
        for power in powers
    ]


def print_row(row: SoftTailRow) -> None:
    print(
        f"n={row.n} r={row.period} odd=({row.period_odd_part},"
        f"{row.next_period_odd_part}) s={row.power:g} C={row.threshold_c:g} "
        f"F={row.estimate:.9g} se={row.standard_error:.3g} "
        f"ESS={row.effective_sample_size:.1f} "
        f"max_share={row.maximum_weight_fraction:.3g} "
        f"active={row.active_fraction:.6f} time={row.seconds:.3f}s",
        flush=True,
    )


def write_csv(path: Path, rows: list[SoftTailRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(SoftTailRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--periods", required=True)
    parser.add_argument("--threshold-c", type=float, default=2.0)
    parser.add_argument("--powers", default="1")
    parser.add_argument("--sample-count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-chunk-size", type=int, default=1024)
    parser.add_argument("--residue-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    periods = parse_integer_list(args.periods)
    powers = parse_float_list(args.powers)
    rng = np.random.default_rng(args.seed)
    output_bits = uniform_output_bits(args.n, args.sample_count, rng)
    rows: list[SoftTailRow] = []
    for period in periods:
        period_rows = estimate_period(
            output_bits,
            period,
            threshold_c=args.threshold_c,
            powers=powers,
            output_chunk_size=args.output_chunk_size,
            residue_chunk_size=args.residue_chunk_size,
        )
        rows.extend(period_rows)
        for row in period_rows:
            print_row(row)

    if args.output is not None:
        write_csv(args.output, rows)
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
