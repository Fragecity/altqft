#!/usr/bin/env python3
"""Plot fixed- and moving-period HP-1 active-tail subset simulations."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Point:
    nqubit: int
    period: int
    log_active_fraction: float

    @property
    def log_normalized_fraction(self) -> float:
        return self.log_active_fraction - 2.0 * math.log(float(self.period))


@dataclass(frozen=True)
class Fit:
    slope: float
    intercept: float
    r_squared: float


def inside_restricted_window(point: Point) -> bool:
    """Test r < 2**(n/4) in log space without overflowing."""
    return math.log2(float(point.period)) < point.nqubit / 4.0


def load_points(path: Path, period: int | None = None) -> list[Point]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = [
        Point(
            nqubit=int(row["n"]),
            period=int(row["period"]),
            log_active_fraction=float(row["log_active_fraction"]),
        )
        for row in rows
        if period is None or int(row["period"]) == period
    ]
    return sorted(
        (point for point in points if inside_restricted_window(point)),
        key=lambda point: (point.nqubit, point.period),
    )


def linear_fit(points: list[Point]) -> Fit:
    x_values = np.asarray([point.nqubit for point in points], dtype=np.float64)
    y_values = np.asarray(
        [point.log_normalized_fraction for point in points],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(x_values, y_values, 1)
    prediction = slope * x_values + intercept
    residual_sum = float(np.square(y_values - prediction).sum())
    total_sum = float(np.square(y_values - y_values.mean()).sum())
    return Fit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=1.0 - residual_sum / total_sum,
    )


def fit_label(name: str, fit: Fit) -> str:
    return (
        name
        + rf": $y={fit.slope:.4f}n{fit.intercept:+.3f}$"
        + rf", $R^2={fit.r_squared:.4f}$"
    )


def write_displayed_points(path: Path, series: list[tuple[str, list[Point]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["series", "n", "period", "log_active_fraction", "log_G_tau"]
        )
        for name, points in series:
            for point in points:
                writer.writerow(
                    [
                        name,
                        point.nqubit,
                        point.period,
                        f"{point.log_active_fraction:.17g}",
                        f"{point.log_normalized_fraction:.17g}",
                    ]
                )


def plot(
    series: list[tuple[str, list[Point]]],
    output_paths: list[Path],
    conservative_beta: float,
) -> None:
    styles = [
        ("#08519C", "o", "-"),
        ("#238B45", "s", "-"),
        ("#CB181D", "^", "-"),
        ("#6A51A3", "D", "-."),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.3,
        }
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)

    all_n: list[int] = []
    fits: list[tuple[str, Fit]] = []
    for (name, points), (color, marker, linestyle) in zip(series, styles):
        if len(points) < 2:
            raise ValueError(f"series {name!r} has fewer than two in-window points")
        fit = linear_fit(points)
        fits.append((name, fit))
        x_values = np.asarray([point.nqubit for point in points], dtype=np.float64)
        y_values = np.asarray(
            [point.log_normalized_fraction for point in points], dtype=np.float64
        )
        all_n.extend(int(value) for value in x_values)
        axis.scatter(
            x_values,
            y_values,
            s=36,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        line_x = np.linspace(float(x_values.min()), float(x_values.max()), 300)
        axis.plot(
            line_x,
            fit.slope * line_x + fit.intercept,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=fit_label(name, fit),
        )

    reference_x = np.linspace(float(min(all_n)), float(max(all_n)), 300)
    axis.plot(
        reference_x,
        -conservative_beta * reference_x,
        color="0.30",
        linestyle="--",
        linewidth=1.3,
        label=rf"conservative reference $-{conservative_beta:.2f}n$",
    )
    axis.set_title(r"HP-1 active-tail simulations inside $r<2^{n/4}$", loc="left")
    axis.set_xlabel(r"Number of qubits $n$")
    axis.set_ylabel(r"$\ln G_\tau(n,r)$")
    axis.set_xlim(min(all_n) - 5, max(all_n) + 5)
    axis.grid(True, color="0.88", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="0.85")
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=300 if output_path.suffix.lower() == ".png" else None,
        )
    plt.close(figure)

    for name, fit in fits:
        print(
            f"{name}: slope={fit.slope:.12g} intercept={fit.intercept:.12g} "
            f"R2={fit.r_squared:.12g}"
        )
    for output_path in output_paths:
        print(f"plot={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed",
        type=Path,
        default=Path(
            "data/hp1_active_tail_subset_cuda/r12_r97_valid_n10_180_single.csv"
        ),
    )
    parser.add_argument(
        "--r53",
        type=Path,
        default=Path(
            "data/hp1_active_tail_subset_cuda/r53_valid_n30_180_single.csv"
        ),
    )
    parser.add_argument(
        "--moving",
        type=Path,
        default=Path(
            "data/hp1_active_tail_subset_cuda/near_n_over_5_valid_n20_80_single.csv"
        ),
    )
    parser.add_argument("--conservative-beta", type=float, default=0.55)
    parser.add_argument(
        "--displayed-csv",
        type=Path,
        default=Path(
            "data/hp1_active_tail_subset_cuda/combined_valid_r12_r53_r97_n_over_5.csv"
        ),
    )
    parser.add_argument("--output", type=Path, action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = [
        (r"$r=12$", load_points(args.fixed, period=12)),
        (r"$r=53$", load_points(args.r53, period=53)),
        (r"$r=97$", load_points(args.fixed, period=97)),
        (r"$r\simeq2^{n/5}$", load_points(args.moving)),
    ]
    output_paths = args.output or [
        Path("figs/fi_fig/hp1_active_tail_multi_period_n_over_5.pdf"),
        Path("figs/fi_fig/hp1_active_tail_multi_period_n_over_5.png"),
    ]
    write_displayed_points(args.displayed_csv, series)
    plot(series, output_paths, args.conservative_beta)
    print(f"displayed_csv={args.displayed_csv}")


if __name__ == "__main__":
    main()
