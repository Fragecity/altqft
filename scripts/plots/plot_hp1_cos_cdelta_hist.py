#!/usr/bin/env python3
"""Plot the CUDA histogram for cos(pi/2 (C Delta)_a)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hp1_cos_cdelta_hist/n100_samples10000_hist.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs/fi_fig/hp1_cos_cdelta_hist_n100_samples10000.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("figs/fi_fig/hp1_cos_cdelta_hist_n100_samples10000.pdf"),
    )
    return parser.parse_args()


def read_histogram(path: Path) -> tuple[list[float], list[float], float]:
    centers: list[float] = []
    probabilities: list[float] = []
    width = 0.0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            left = float(row["bin_left"])
            right = float(row["bin_right"])
            centers.append(float(row["bin_center"]))
            probabilities.append(float(row["probability"]))
            width = right - left
    return centers, probabilities, width


def main() -> None:
    args = parse_args()
    centers, probabilities, width = read_histogram(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        centers,
        probabilities,
        width=0.98 * width,
        color="#2f6f9f",
        edgecolor="#1f4059",
        linewidth=0.25,
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel(r"$\cos\{\pi (C\Delta_{q,q',0,0})_a/2\}$")
    ax.set_ylabel("Probability")
    ax.set_title(r"HP-1 $C\Delta$ cosine statistic, $n=100$, 10000 random $0<q<q'<R_s$")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.pdf_output)
    print(f"wrote {args.output}")
    print(f"wrote {args.pdf_output}")


if __name__ == "__main__":
    main()
