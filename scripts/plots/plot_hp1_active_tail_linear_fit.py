#!/usr/bin/env python3
"""Plot the log-linear fits supporting the HP-1 active-tail count law."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r_squared: float


@dataclass(frozen=True)
class SubsetPoint:
    nqubit: int
    period: int
    log_normalized_fraction: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def linear_fit(x_values: list[float], y_values: list[float]) -> LinearFit:
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    slope, intercept = np.polyfit(x_array, y_array, 1)
    prediction = slope * x_array + intercept
    residual_sum = float(np.square(y_array - prediction).sum())
    total_sum = float(np.square(y_array - y_array.mean()).sum())
    return LinearFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=1.0 - residual_sum / total_sum,
    )


def load_exact_points(
    path: Path,
    fit_n_min: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]], LinearFit]:
    all_points: list[tuple[int, float]] = []
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in read_rows(path):
        point = (int(row["n"]), float(row["log_normalized_fraction"]))
        all_points.append(point)
        grouped[point[0]].append(point[1])

    envelope = [(nqubit, min(values)) for nqubit, values in sorted(grouped.items())]
    fit_points = [point for point in envelope if point[0] >= fit_n_min]
    fit = linear_fit(
        [float(point[0]) for point in fit_points],
        [point[1] for point in fit_points],
    )
    return all_points, fit_points, fit


def load_subset_points(
    paths: list[Path],
    period: int | None = None,
) -> list[SubsetPoint]:
    points = [
        SubsetPoint(
            nqubit=int(row["n"]),
            period=int(row["period"]),
            log_normalized_fraction=(
                float(row["log_active_fraction"])
                - 2.0 * math.log(float(row["period"]))
            ),
        )
        for path in paths
        for row in read_rows(path)
        if period is None or int(row["period"]) == period
    ]
    return sorted(points, key=lambda point: (point.nqubit, point.period))


def fit_subset_points(points: list[SubsetPoint]) -> LinearFit:
    return linear_fit(
        [float(point.nqubit) for point in points],
        [point.log_normalized_fraction for point in points],
    )


def format_fit(fit: LinearFit, slope_digits: int) -> str:
    return (
        rf"$y={fit.slope:.{slope_digits}f}n{fit.intercept:+.3f}$"
        + "\n"
        + rf"$R^2={fit.r_squared:.4f}$"
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def plot_fits(
    exact_path: Path,
    subset_paths: list[Path],
    edge_subset_path: Path,
    output_paths: list[Path],
    *,
    fit_n_min: int,
    period: int,
    conservative_beta: float,
) -> None:
    exact_points, envelope, exact_fit = load_exact_points(exact_path, fit_n_min)
    subset_points = load_subset_points(subset_paths, period)
    edge_points = load_subset_points([edge_subset_path])
    subset_fit = fit_subset_points(subset_points)
    edge_fit = fit_subset_points(edge_points)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.3,
        }
    )
    figure, (exact_axis, subset_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        constrained_layout=True,
    )

    light_blue = "#9ECAE1"
    medium_blue = "#3182BD"
    dark_blue = "#08519C"
    deep_blue = "#08306B"
    exact_axis.scatter(
        [point[0] for point in exact_points],
        [point[1] for point in exact_points],
        s=13,
        color=light_blue,
        alpha=0.75,
        linewidths=0,
        label="_nolegend_",
    )
    exact_axis.scatter(
        [point[0] for point in envelope],
        [point[1] for point in envelope],
        s=28,
        color=medium_blue,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="_nolegend_",
    )
    exact_line_x = np.linspace(float(envelope[0][0]), float(envelope[-1][0]), 200)
    exact_axis.plot(
        exact_line_x,
        exact_fit.slope * exact_line_x + exact_fit.intercept,
        color=deep_blue,
        linewidth=1.6,
        label=format_fit(exact_fit, slope_digits=3),
    )
    exact_axis.plot(
        exact_line_x,
        -conservative_beta * exact_line_x,
        color=dark_blue,
        linestyle="--",
        linewidth=1.2,
        label="_nolegend_",
    )
    exact_axis.set_title("(a) Exact restricted-window enumeration", loc="left")
    exact_axis.set_xlabel(r"Number of qubits $n$")
    exact_axis.set_ylabel(r"$\ln G_\tau(n,r)$")
    exact_axis.set_xticks([14, 16, 18, 20])
    exact_axis.set_xlim(13.65, 20.35)
    exact_axis.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
        handlelength=2.2,
    )
    style_axis(exact_axis)

    subset_x = [point.nqubit for point in subset_points]
    subset_y = [point.log_normalized_fraction for point in subset_points]
    edge_x = [point.nqubit for point in edge_points]
    edge_y = [point.log_normalized_fraction for point in edge_points]
    subset_axis.scatter(
        subset_x,
        subset_y,
        s=25,
        color=medium_blue,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="_nolegend_",
    )
    subset_line_x = np.linspace(float(min(subset_x)), float(max(subset_x)), 200)
    subset_axis.plot(
        subset_line_x,
        subset_fit.slope * subset_line_x + subset_fit.intercept,
        color=deep_blue,
        linewidth=1.6,
        label=format_fit(subset_fit, slope_digits=4),
    )
    subset_axis.scatter(
        edge_x,
        edge_y,
        s=31,
        marker="^",
        color=light_blue,
        edgecolors=dark_blue,
        linewidths=0.6,
        zorder=4,
        label="_nolegend_",
    )
    edge_line_x = np.linspace(float(min(edge_x)), float(max(edge_x)), 100)
    subset_axis.plot(
        edge_line_x,
        edge_fit.slope * edge_line_x + edge_fit.intercept,
        color=dark_blue,
        linestyle="--",
        linewidth=1.5,
        label=format_fit(edge_fit, slope_digits=4),
    )
    fixed_label_n = 118.0
    subset_axis.text(
        fixed_label_n,
        subset_fit.slope * fixed_label_n + subset_fit.intercept + 2.0,
        rf"$r={period}$",
        color=deep_blue,
        fontsize=8,
    )
    edge_label_n = 51.0
    subset_axis.text(
        edge_label_n,
        edge_fit.slope * edge_label_n + edge_fit.intercept - 1.0,
        r"$r\simeq 2^{n/4}$",
        color=dark_blue,
        fontsize=8,
    )
    subset_axis.set_title("(b) Subset simulations across periods", loc="left")
    subset_axis.set_xlabel(r"Number of qubits $n$")
    subset_axis.set_ylabel(r"$\ln G_\tau(n,r)$")
    subset_axis.set_xticks([20, 50, 80, 110, 140, 170, 200])
    subset_axis.set_xlim(14, 206)
    subset_axis.legend(loc="upper right", frameon=False, handlelength=2.2)
    style_axis(subset_axis)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=300 if output_path.suffix.lower() == ".png" else None,
        )
    plt.close(figure)

    print(
        "exact fit: "
        f"slope={exact_fit.slope:.9f}, intercept={exact_fit.intercept:.9f}, "
        f"R2={exact_fit.r_squared:.9f}"
    )
    print(
        "subset fit: "
        f"slope={subset_fit.slope:.9f}, intercept={subset_fit.intercept:.9f}, "
        f"R2={subset_fit.r_squared:.9f}"
    )
    print(
        "edge fit: "
        f"slope={edge_fit.slope:.9f}, intercept={edge_fit.intercept:.9f}, "
        f"R2={edge_fit.r_squared:.9f}"
    )
    for output_path in output_paths:
        print(f"plot={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot active-tail exact-envelope and subset-simulation linear fits.",
    )
    parser.add_argument(
        "--exact",
        type=Path,
        default=Path("data/hp1_active_tail_exact/even_non_dyadic_n10_20.csv"),
    )
    parser.add_argument(
        "--subset",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--edge-subset",
        type=Path,
        default=Path(
            "data/hp1_active_tail_subset_cuda/near_window_edge_n20_60_single.csv"
        ),
    )
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--fit-n-min", type=int, default=14)
    parser.add_argument("--conservative-beta", type=float, default=0.55)
    parser.add_argument("--output", type=Path, action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_paths = args.subset or [
        Path("data/hp1_active_tail_subset_cuda/r12_n20_200_step10_single.csv"),
    ]
    output_paths = args.output or [
        Path("figs/fi_fig/hp1_active_tail_linear_fits.pdf"),
        Path("figs/fi_fig/hp1_active_tail_linear_fits.png"),
    ]
    plot_fits(
        args.exact,
        subset_paths,
        args.edge_subset,
        output_paths,
        fit_n_min=args.fit_n_min,
        period=args.period,
        conservative_beta=args.conservative_beta,
    )


if __name__ == "__main__":
    main()
