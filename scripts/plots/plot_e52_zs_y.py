#!/usr/bin/env python3
"""Plot y = |Z_s(x)|^2 / R^(s) from the CUDA E52 experiment CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def load_by_n(path: Path) -> dict[int, list[float]]:
    by_n: dict[int, list[float]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            by_n[int(row["n"])].append(float(row["y"]))
    return dict(sorted(by_n.items()))


def write_summary(path: Path, by_n: dict[int, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, values in by_n.items():
        sorted_values = sorted(values)
        rows.append(
            {
                "n": n,
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p10": percentile(sorted_values, 0.10),
                "p90": percentile(sorted_values, 0.90),
                "min": sorted_values[0],
                "max": sorted_values[-1],
            }
        )
    path.write_text(json.dumps(rows, indent=2) + "\n")


def plot(input_path: Path, output_paths: list[Path], summary_path: Path | None) -> None:
    by_n = load_by_n(input_path)
    if summary_path is not None:
        write_summary(summary_path, by_n)

    ns = list(by_n)
    means = [statistics.fmean(by_n[n]) for n in ns]
    medians = [statistics.median(by_n[n]) for n in ns]
    p10s = [percentile(sorted(by_n[n]), 0.10) for n in ns]
    p90s = [percentile(sorted(by_n[n]), 0.90) for n in ns]

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    for n in ns:
        values = by_n[n]
        ax.scatter(
            [n] * len(values),
            values,
            s=5,
            alpha=0.055,
            color="#4c78a8",
            linewidths=0,
        )
    ax.fill_between(ns, p10s, p90s, color="#9ecae1", alpha=0.35, label="10-90%")
    ax.plot(ns, means, color="#1f77b4", linewidth=1.7, marker="o", markersize=3.5, label="mean")
    ax.plot(ns, medians, color="black", linewidth=1.4, marker="s", markersize=3.0, label="median")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$y=|Z_s(x)|^2/R^{(s)}$")
    ax.set_xticks(ns)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/e52_zs_cuda/e52_zs_y.csv"))
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
