#!/usr/bin/env python3
"""Estimate how many fixed-phase HP-1 outputs have exponentially small Pr(x).

For a threshold ``Pr_r(x) <= C / 2**n``, the reported quantity is

    |Omega_C| / 2**n
      = E_{X uniform}[1{2**n Pr_r(X) <= C}].

The estimator samples output strings uniformly; it does not sample from the
HP-1 output law.  Point probabilities are evaluated with a roots-of-unity
filter, so no ``2**n`` state vector is constructed.
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
    dyadic_zero_fraction_lower_bound,
    hoeffding_radius,
    hp1_log2_scaled_probabilities,
    normalization_fraction_lower_bound,
    small_probability_fraction,
    two_adic_parts,
    uniform_output_bits,
)


@dataclass(frozen=True)
class Threshold:
    label: str
    log2_scaled_value: float


@dataclass(frozen=True)
class FractionRow:
    n: int
    period: int
    two_adic_power: int
    odd_part: int
    threshold: str
    log2_scaled_threshold: float
    sample_count: int
    small_count: int
    fraction_estimate: float
    confidence: float
    confidence_lower: float
    confidence_upper: float
    normalization_lower_bound: float
    dyadic_zero_lower_bound: float
    analytic_lower_bound: float
    seconds: float


def parse_number_list(raw_value: str) -> list[float]:
    values = [float(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one number")
    return values


def parse_integer_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def build_thresholds(
    nqubit: int,
    c_values: list[float],
    beta_values: list[float],
) -> list[Threshold]:
    thresholds: list[Threshold] = []
    for value in c_values:
        if value <= 0.0:
            raise ValueError("C values must be positive")
        thresholds.append(Threshold(label=f"C={value:g}", log2_scaled_value=math.log2(value)))
    for value in beta_values:
        thresholds.append(
            Threshold(
                label=f"beta={value:g}",
                log2_scaled_value=(1.0 - value) * nqubit,
            )
        )
    return thresholds


def estimate_period(
    nqubit: int,
    period: int,
    output_bits: np.ndarray,
    thresholds: list[Threshold],
    *,
    confidence: float,
    output_chunk_size: int,
    residue_chunk_size: int,
) -> list[FractionRow]:
    started = time.perf_counter()
    log2_scaled = hp1_log2_scaled_probabilities(
        output_bits,
        period,
        output_chunk_size=output_chunk_size,
        residue_chunk_size=residue_chunk_size,
    )
    elapsed = time.perf_counter() - started
    sample_count = output_bits.shape[0]
    radius = hoeffding_radius(sample_count, confidence)
    two_power, odd_part = two_adic_parts(period)
    rows: list[FractionRow] = []

    for threshold in thresholds:
        fraction = small_probability_fraction(log2_scaled, threshold.log2_scaled_value)
        small_count = int(np.count_nonzero(log2_scaled <= threshold.log2_scaled_value))
        normalization_bound = normalization_fraction_lower_bound(
            threshold.log2_scaled_value
        )
        dyadic_zero_bound = dyadic_zero_fraction_lower_bound(nqubit, period)
        rows.append(
            FractionRow(
                n=nqubit,
                period=period,
                two_adic_power=two_power,
                odd_part=odd_part,
                threshold=threshold.label,
                log2_scaled_threshold=threshold.log2_scaled_value,
                sample_count=sample_count,
                small_count=small_count,
                fraction_estimate=fraction,
                confidence=confidence,
                confidence_lower=max(0.0, fraction - radius),
                confidence_upper=min(1.0, fraction + radius),
                normalization_lower_bound=normalization_bound,
                dyadic_zero_lower_bound=dyadic_zero_bound,
                analytic_lower_bound=max(normalization_bound, dyadic_zero_bound),
                seconds=elapsed,
            )
        )
    return rows


def write_csv(path: Path, rows: list[FractionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(FractionRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def print_row(row: FractionRow) -> None:
    print(
        f"n={row.n} r={row.period} (2^{row.two_adic_power}*{row.odd_part}) "
        f"{row.threshold}: |Omega|/2^n={row.fraction_estimate:.6f} "
        f"CI=[{row.confidence_lower:.6f},{row.confidence_upper:.6f}] "
        f"analytic>={row.analytic_lower_bound:.6f} "
        f"M={row.sample_count} time={row.seconds:.3f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--periods", required=True, help="Comma-separated periods r.")
    parser.add_argument(
        "--c-values",
        default="1,2,4",
        help="Thresholds C in Pr_r(x) <= C/2^n; use an empty string to disable.",
    )
    parser.add_argument(
        "--beta-values",
        default="",
        help="Threshold exponents beta in Pr_r(x) <= 2^(-beta*n).",
    )
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-chunk-size", type=int, default=256)
    parser.add_argument("--residue-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_values = parse_number_list(args.c_values) if args.c_values.strip() else []
    beta_values = parse_number_list(args.beta_values) if args.beta_values.strip() else []
    thresholds = build_thresholds(args.n, c_values, beta_values)
    if not thresholds:
        raise ValueError("at least one C or beta threshold is required")

    periods = parse_integer_list(args.periods)
    rng = np.random.default_rng(args.seed)
    output_bits = uniform_output_bits(args.n, args.sample_count, rng)
    rows: list[FractionRow] = []
    for period in periods:
        period_rows = estimate_period(
            args.n,
            period,
            output_bits,
            thresholds,
            confidence=args.confidence,
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
