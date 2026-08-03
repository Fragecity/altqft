#!/usr/bin/env python3
"""Plot calibrated HP-1 C_1/2 minima near r ~= 2^(n/2)."""

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
class CalibrationPoint:
    n: int
    c_min: float
    method: str


CALIBRATION = [
    CalibrationPoint(8, 0.2272116043720387, "exact"),
    CalibrationPoint(10, 0.30595338122738536, "exact"),
    CalibrationPoint(12, 0.3497852312729033, "exact"),
    CalibrationPoint(14, 0.37233136596986594, "exact"),
    CalibrationPoint(16, 0.39834109933175554, "exact"),
    CalibrationPoint(18, 0.4166048129858444, "exact"),
    CalibrationPoint(20, 0.4630572120192964, "exact"),
    CalibrationPoint(22, 0.47486900031762147, "exact"),
    CalibrationPoint(24, 0.5083317905849727, "exact"),
    CalibrationPoint(26, 0.5289146329774805, "exact"),
    CalibrationPoint(28, 0.565928884858609, "exact"),
    CalibrationPoint(32, 0.6024713796610133, "mc-is"),
]


def rounded_sqrt_power_two(n: int) -> int:
    value = 1 << n
    lower = math.isqrt(value)
    upper = lower + 1
    return upper if upper * upper - value < value - lower * lower else lower


def fit_log_model(points: list[CalibrationPoint]) -> tuple[float, float, float]:
    x_values = np.log(np.array([point.n for point in points], dtype=np.float64))
    y_values = np.array([point.c_min for point in points], dtype=np.float64)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    prediction = slope * x_values + intercept
    r_squared = 1.0 - float(np.square(y_values - prediction).sum()) / float(
        np.square(y_values - y_values.mean()).sum()
    )
    return float(slope), float(intercept), r_squared


def write_estimates(
    path: Path,
    *,
    n_min: int,
    n_max: int,
    window_radius: int,
    slope: float,
    intercept: float,
    r_squared: float,
) -> list[dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for n_value in range(n_min, n_max + 1):
        center = rounded_sqrt_power_two(n_value)
        estimate = slope * math.log(n_value) + intercept
        rows.append(
            {
                "n": str(n_value),
                "center_r": str(center),
                "period_min": str(max(2, center - window_radius)),
                "period_max": str(center + window_radius),
                "window_radius": str(window_radius),
                "c_half_min_estimate": f"{estimate:.12g}",
                "model": "log_calibrated_exact_mc",
                "fit_slope": f"{slope:.12g}",
                "fit_intercept": f"{intercept:.12g}",
                "fit_r_squared": f"{r_squared:.12g}",
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_calibration(path: Path, *, window_radius: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["n", "c_half_min", "method", "window_radius"],
        )
        writer.writeheader()
        for point in CALIBRATION:
            writer.writerow(
                {
                    "n": point.n,
                    "c_half_min": f"{point.c_min:.12g}",
                    "method": point.method,
                    "window_radius": window_radius,
                }
            )


def plot_estimates(
    rows: list[dict[str, str]],
    output_paths: list[Path],
    *,
    slope: float,
    intercept: float,
    r_squared: float,
) -> None:
    n_values = np.array([int(row["n"]) for row in rows], dtype=np.float64)
    estimates = np.array(
        [float(row["c_half_min_estimate"]) for row in rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        n_values,
        estimates,
        color="#1f77b4",
        linewidth=2.0,
        label=rf"${slope:.3f}\log n {intercept:+.3f}$, $R^2={r_squared:.3f}$",
    )
    ax.set_xlabel(r"Number of qubits $n$")
    ax.set_ylabel(r"local minimum $C_{1/2}$")
    ax.set_title(r"Fixed-phase HP-1 $C_{1/2}$ near $r=2^{n/2}$")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240 if output_path.suffix == ".png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-min", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=250)
    parser.add_argument("--window-radius", type=int, default=16)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/hp1_chernoff/sqrt_window_min_n50_250_estimate.csv"),
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=Path("data/hp1_chernoff/sqrt_window_min_calibration.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        action="append",
        default=[
            Path("figs/fi_fig/hp1_chernoff_sqrt_window_min_n50_250.png"),
            Path("figs/fi_fig/hp1_chernoff_sqrt_window_min_n50_250.pdf"),
        ],
    )
    args = parser.parse_args()

    slope, intercept, r_squared = fit_log_model(CALIBRATION)
    rows = write_estimates(
        args.csv_output,
        n_min=args.n_min,
        n_max=args.n_max,
        window_radius=args.window_radius,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
    )
    write_calibration(args.calibration_output, window_radius=args.window_radius)
    plot_estimates(rows, args.plot_output, slope=slope, intercept=intercept, r_squared=r_squared)

    min_row = min(rows, key=lambda row: float(row["c_half_min_estimate"]))
    max_row = max(rows, key=lambda row: float(row["c_half_min_estimate"]))
    print(f"csv={args.csv_output}")
    print(f"calibration={args.calibration_output}")
    for output_path in args.plot_output:
        print(f"plot={output_path}")
    print(
        "estimate range: "
        f"n={min_row['n']} C={min_row['c_half_min_estimate']} to "
        f"n={max_row['n']} C={max_row['c_half_min_estimate']}"
    )


if __name__ == "__main__":
    main()
