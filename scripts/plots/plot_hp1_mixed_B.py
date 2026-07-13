#!/usr/bin/env python3
"""Plot the HP-1 mixed omega sum against the dyadic exponent a."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def guessed_kappa(a_value: int) -> float:
    if a_value == 0:
        return 1.0
    return 0.95 if a_value % 2 == 0 else 0.72


def load_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            n_value = int(row["n"])
            a_value = int(row["a"])
            u_value = int(row["u"]) if row.get("u") else 0
            v_value = int(row["v"]) if row.get("v") else 0
            s_value = int(row["s"]) if row.get("s") else 0
            t_value = int(row["t"]) if row.get("t") else 0

            if row.get("B_over_2_n_plus_a_minus_1"):
                b_value = float(row["B_over_2_n_plus_a_minus_1"]) * (2.0**a_value)
            elif row.get("mixed_sum"):
                b_value = float(row["mixed_sum"]) / (2.0 ** (n_value - 1))
            else:
                scaled = float(row["scaled"])
                b_value = scaled / (float(s_value) * float(t_value))

            if s_value and t_value:
                guess_value = guessed_kappa(a_value) * (2.0**a_value) / (
                    float(s_value) * float(t_value)
                )
            elif u_value and v_value:
                guess_value = guessed_kappa(a_value) / (
                    float(u_value) * float(v_value) * (2.0**a_value)
                )
            else:
                raise ValueError("input rows need either s,t or u,v columns")

            if b_value <= 0.0 or guess_value <= 0.0:
                continue
            rows.append(
                {
                    "n": n_value,
                    "a": a_value,
                    "u": u_value,
                    "v": v_value,
                    "s": s_value,
                    "t": t_value,
                    "B": b_value,
                    "guess": guess_value,
                }
            )
    return rows


def median_by_a(rows: list[dict[str, float | int]], key: str) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["a"])].append(float(row[key]))
    a_values = sorted(grouped)
    return a_values, [median(grouped[a_value]) for a_value in a_values]


def plot(input_path: Path, output_paths: list[Path]) -> None:
    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"no rows loaded from {input_path}")

    n_values = sorted({int(row["n"]) for row in rows})
    if len(n_values) != 1:
        raise ValueError("plot expects one fixed n value")
    n_value = n_values[0]

    xs = [int(row["a"]) for row in rows]
    ys = [float(row["B"]) for row in rows]
    guess_xs, guess_ys = median_by_a(rows, "guess")
    data_xs, data_medians = median_by_a(rows, "B")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.scatter(xs, ys, s=18, alpha=0.48, linewidths=0, color="#2f6f9f", label="data")
    ax.plot(
        data_xs,
        data_medians,
        color="#2f6f9f",
        linewidth=1.2,
        alpha=0.65,
        label="_nolegend_",
    )
    ax.plot(
        guess_xs,
        guess_ys,
        "--",
        color="black",
        linewidth=1.6,
        label=r"$\kappa_a2^a/(st)$",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_2(\gcd(s,t))$")
    ax.set_ylabel(r"$B$")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hp1_c_twisted_dyadic_resonance_n26_maxa14_30pairs.csv"),
    )
    parser.add_argument("--output", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
