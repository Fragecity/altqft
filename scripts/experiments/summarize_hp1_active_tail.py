#!/usr/bin/env python3
"""Summarize exact and n=200 active-tail HP-1 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    x_values = np.asarray([point[0] for point in points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in points], dtype=np.float64)
    design = np.column_stack((x_values, np.ones_like(x_values)))
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    fitted = slope * x_values + intercept
    residual = float(np.square(y_values - fitted).sum())
    total = float(np.square(y_values - y_values.mean()).sum())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
    }


def exact_envelope(path: Path, fit_n_min: int) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(path):
        by_n[int(row["n"])].append(row)

    envelope: list[dict[str, float | int]] = []
    for nqubit, rows in sorted(by_n.items()):
        minimum = min(rows, key=lambda row: float(row["log_dfi_count_bound"]))
        envelope.append(
            {
                "n": nqubit,
                "period": int(minimum["period"]),
                "active_fraction": float(minimum["active_fraction"]),
                "log_normalized_fraction": float(minimum["log_normalized_fraction"]),
                "log_dfi_count_bound": float(minimum["log_dfi_count_bound"]),
            }
        )
    fit = linear_fit(
        [
            (float(row["n"]), float(row["log_normalized_fraction"]))
            for row in envelope
            if int(row["n"]) >= fit_n_min
        ]
    )
    return envelope, fit


def grouped_subset_points(paths: list[Path], period: int) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for path in paths:
        for row in read_rows(path):
            if int(row["period"]) == period:
                grouped[int(row["n"])].append(float(row["log_active_fraction"]))

    points: list[dict[str, float | int]] = []
    for nqubit, values in sorted(grouped.items()):
        points.append(
            {
                "n": nqubit,
                "replicates": len(values),
                "mean_log_active_fraction": mean(values),
                "sd_log_active_fraction": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    fit = linear_fit(
        [(float(row["n"]), float(row["mean_log_active_fraction"])) for row in points]
    )
    return points, fit


def n200_period_summary(paths: list[Path]) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    for path in paths:
        rows = read_rows(path)
        period = int(rows[0]["period"])
        nqubit = int(rows[0]["n"])
        values = [float(row["log_active_fraction"]) for row in rows]
        normalized = [value - 2.0 * math.log(period) for value in values]
        summaries.append(
            {
                "n": nqubit,
                "period": period,
                "replicates": len(rows),
                "particles": int(rows[0]["particles"]),
                "mutation_steps": int(rows[0]["mutation_steps"]),
                "mean_log_active_fraction": mean(values),
                "sd_log_active_fraction": stdev(values) if len(values) > 1 else 0.0,
                "mean_log_normalized_fraction": mean(normalized),
                "minimum_acceptance": min(float(row["minimum_acceptance"]) for row in rows),
                "minimum_distinct_fraction": min(
                    int(row["final_distinct_states"]) / int(row["particles"])
                    for row in rows
                ),
            }
        )
    return sorted(summaries, key=lambda row: int(row["period"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact",
        type=Path,
        default=Path("data/hp1_active_tail_exact/even_non_dyadic_n10_20.csv"),
    )
    parser.add_argument(
        "--r12-grid",
        type=Path,
        default=Path("data/hp1_active_tail_subset_cuda/r12_n20_180_replicates.csv"),
    )
    parser.add_argument(
        "--r12-n200",
        type=Path,
        default=Path("data/hp1_active_tail_subset_cuda/r12_n200_replicates.csv"),
    )
    parser.add_argument(
        "--n200-representatives",
        type=Path,
        nargs="+",
        default=[
            Path("data/hp1_active_tail_subset_cuda/r12_n200_replicates.csv"),
            Path("data/hp1_active_tail_subset_cuda/r14_n200_heavy.csv"),
            Path("data/hp1_active_tail_subset_cuda/r20_n200_heavy.csv"),
            Path("data/hp1_active_tail_subset_cuda/r24_n200_heavy.csv"),
        ],
    )
    parser.add_argument("--tau", type=float, default=3.0e-4)
    parser.add_argument("--conservative-beta", type=float, default=0.55)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/ai/hp1_active_tail_n200_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envelope, envelope_fit = exact_envelope(args.exact, fit_n_min=14)
    r12_points, r12_fit = grouped_subset_points(
        [args.r12_grid, args.r12_n200],
        period=12,
    )
    representatives = n200_period_summary(args.n200_representatives)

    observed_margins = [
        float(row["log_normalized_fraction"]) + args.conservative_beta * int(row["n"])
        for row in envelope
    ]
    observed_margins.extend(
        float(row["mean_log_normalized_fraction"])
        + args.conservative_beta * int(row["n"])
        for row in representatives
    )
    summary: dict[str, Any] = {
        "status": "conditional statistical law; not a uniform theorem",
        "definition": (
            "A_tau(n,r)={x: 2^n P_r(x)<2 and "
            "(2^n(P_{r+1}(x)-P_r(x)))^2 r^2 >= tau 2^n}"
        ),
        "tau": args.tau,
        "deterministic_point_certificate": "I_r >= tau |A_tau(n,r)|/(2 r^2)",
        "tested_class": "even non-dyadic periods; dyadic periods use exact support mismatch",
        "exact_full_window": {
            "n_range": [min(int(row["n"]) for row in envelope), max(int(row["n"]) for row in envelope)],
            "period_window": "2 <= r < 2^(n/4), even non-dyadic",
            "envelope": envelope,
            "log_normalized_envelope_fit_n14_20": envelope_fit,
        },
        "rare_event_method": {
            "name": "adaptive subset simulation",
            "point_forward": "exact roots-of-unity filter in log domain",
            "validation": {
                "n20_r12_exact_fraction": 0.06952095031738281,
                "n22_r12_exact_fraction": 0.032258033752441406,
                "n30_r12_uniform_mc_fraction": 0.00213275,
            },
            "caveat": (
                "Subset simulation has MCMC mixing error. Independent replicates, "
                "acceptance, and distinct-state diagnostics are reported."
            ),
        },
        "r12_scaling": {
            "points": r12_points,
            "fit_log_active_fraction": r12_fit,
            "fit_log_normalized_fraction": {
                **r12_fit,
                "intercept": r12_fit["intercept"] - 2.0 * math.log(12.0),
            },
        },
        "n200_representative_periods": representatives,
        "conservative_statistical_assumption": {
            "formula": "|A_tau(n,r)|/(2^n r^2) >= exp(-beta n)",
            "beta": args.conservative_beta,
            "minimum_observed_log_margin": min(observed_margins),
            "condition_for_positive_exponent": "beta < ln(2)",
            "ln2": math.log(2.0),
            "implied_dfi_exponent": math.log(2.0) - args.conservative_beta,
            "implied_bound": (
                f"I_r >= {args.tau / 2.0:g} "
                f"exp(({math.log(2.0) - args.conservative_beta:.12g}) n)"
            ),
        },
        "unresolved_scope": (
            "Uniformity over every even non-dyadic r at n=200 is an explicit "
            "statistical assumption; point queries with exponentially large odd "
            "parts are not evaluated. Odd denominator periods remain separate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["r12_scaling"]["fit_log_active_fraction"], indent=2))
    print(json.dumps(summary["conservative_statistical_assumption"], indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
