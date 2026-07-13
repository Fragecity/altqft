#!/usr/bin/env python3
"""Plot the large-n HP-1 Chernoff coefficient estimates produced by the CUDA sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REGIME_ORDER = ["2^(n/2)", "2^(n/4)", "n^2", "n^3"]
COLORS = {
    "2^(n/2)": "#1f77b4",
    "2^(n/4)": "#2ca02c",
    "n^2": "#d62728",
    "n^3": "#9467bd",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot(input_path: Path, output_paths: list[Path], *, metric: str) -> None:
    rows = load_rows(input_path)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    if metric == "coefficient":
        value_key = "coefficient"
        error_key = "coefficient_se"
        ylabel = r"$C(\Pr_r,\Pr_{r+1})$"
    elif metric == "information":
        value_key = "chernoff"
        error_key = "chernoff_se"
        ylabel = r"$-\ln C(\Pr_r,\Pr_{r+1})$"
    else:
        raise ValueError(f"unknown metric: {metric}")

    for regime in REGIME_ORDER:
        points = sorted(
            (row for row in rows if row["regime"] == regime),
            key=lambda row: int(row["n"]),
        )
        if not points:
            continue
        x_values = [int(row["n"]) for row in points]
        y_values = [float(row[value_key]) for row in points]
        methods = {row.get("method", "") for row in points}
        if "cuda_mc_uniform_alpha_half" in methods:
            y_errors = [float(row.get(error_key, "0") or 0.0) for row in points]
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                marker="o",
                markersize=4,
                linewidth=1.5,
                elinewidth=0.8,
                capsize=2,
                label=f"{regime} MC",
                color=COLORS[regime],
            )
            continue
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            label=f"{regime} analytic",
            color=COLORS[regime],
        )
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    if metric == "coefficient":
        ax.set_yscale("log")
    ax.grid(True, color="0.88", linewidth=0.7)
    if metric == "coefficient":
        ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.legend(frameon=False)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hp1_chernoff/four_regimes_cuda_n20_200_hi.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        default=[
            Path("figs/fi_fig/hp1_chernoff_coefficient_four_regimes_cuda.png"),
            Path("figs/fi_fig/hp1_chernoff_coefficient_four_regimes_cuda.pdf"),
        ],
    )
    parser.add_argument(
        "--metric",
        choices=["coefficient", "information"],
        default="coefficient",
    )
    args = parser.parse_args()
    plot(args.input, args.output, metric=args.metric)


if __name__ == "__main__":
    main()
