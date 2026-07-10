#!/usr/bin/env python3
"""Plot the normalized HP-1 overlap factor A(s,t) from CUDA CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_points(path: Path, x_mode: str) -> tuple[list[float], list[float], list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    colors: list[int] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            a_value = int(row["a"])
            y_value = float(row["A"])
            s_value = float(row["s"])
            t_value = float(row["t"])
            if x_mode == "both":
                xs.extend([s_value, t_value])
                ys.extend([y_value, y_value])
                colors.extend([a_value, a_value])
            elif x_mode == "min":
                xs.append(min(s_value, t_value))
                ys.append(y_value)
                colors.append(a_value)
            else:
                xs.append(s_value)
                ys.append(y_value)
                colors.append(a_value)
    return xs, ys, colors


def plot(input_path: Path, output_paths: list[Path], x_mode: str, xscale: str) -> None:
    xs, ys, colors = load_points(input_path, x_mode)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    scatter = ax.scatter(
        xs,
        ys,
        c=colors,
        cmap="viridis",
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$A(s,t)$")
    ax.set_xscale(xscale)
    ax.grid(True, color="0.88", linewidth=0.7)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$a(s,t)$")

    if x_mode == "both":
        ax.set_title(r"$n=30$, 1000 random symmetric $(s,t)$ pairs")
    else:
        ax.set_title(r"$n=30$, 1000 random $(s,t)$ pairs")

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hp1_overlap_a_cuda/n30_random_pairs.csv"),
    )
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--x-mode", choices=("both", "min", "s"), default="both")
    parser.add_argument("--xscale", choices=("linear", "log"), default="log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.input, args.output, args.x_mode, args.xscale)


if __name__ == "__main__":
    main()
