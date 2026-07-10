#!/usr/bin/env python3
"""Scatter P(cos(pi/2 (C Delta)_a) = 0) versus nu_2((q'-q)s)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hp1_cos_cdelta_hist/n100_samples10000_samples.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs/fi_fig/hp1_cos_zero_vs_alpha_n100_samples10000.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("figs/fi_fig/hp1_cos_zero_vs_alpha_n100_samples10000.pdf"),
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--x-max", type=int, default=12)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--render", choices=("scatter", "density"), default="density")
    parser.add_argument("--splat-x", type=float, default=0.42)
    parser.add_argument("--splat-y", type=float, default=0.006)
    parser.add_argument("--grid-x", type=int, default=900)
    parser.add_argument("--grid-y", type=int, default=520)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=0.55)
    return parser.parse_args()


def read_samples(path: Path) -> tuple[list[int], list[float]]:
    alphas: list[int] = []
    p_zero: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alphas.append(int(row["alpha"]))
            p_zero.append(float(row["p_zero"]))
    return alphas, p_zero


def gaussian_splat_density(
    xs: list[int],
    ys: list[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_x: int,
    grid_y: int,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    density = np.zeros((grid_y, grid_x), dtype=np.float64)
    x0, x1 = xlim
    y0, y1 = ylim
    dx = (x1 - x0) / (grid_x - 1)
    dy = (y1 - y0) / (grid_y - 1)
    radius_x = max(1, int(np.ceil(3.0 * sigma_x / dx)))
    radius_y = max(1, int(np.ceil(3.0 * sigma_y / dy)))

    for x_value, y_value in zip(xs, ys):
        ix = int(round((x_value - x0) / dx))
        iy = int(round((y_value - y0) / dy))
        if ix < 0 or ix >= grid_x or iy < 0 or iy >= grid_y:
            continue

        x_left = max(0, ix - radius_x)
        x_right = min(grid_x, ix + radius_x + 1)
        y_bottom = max(0, iy - radius_y)
        y_top = min(grid_y, iy + radius_y + 1)

        x_offsets = (np.arange(x_left, x_right) - ix) * dx
        y_offsets = (np.arange(y_bottom, y_top) - iy) * dy
        gx = np.exp(-0.5 * (x_offsets / sigma_x) ** 2)
        gy = np.exp(-0.5 * (y_offsets / sigma_y) ** 2)
        density[y_bottom:y_top, x_left:x_right] += np.outer(gy, gx)

    return density


def main() -> None:
    args = parse_args()
    alphas, p_zero = read_samples(args.input)
    plotted = [(alpha, value) for alpha, value in zip(alphas, p_zero) if alpha <= args.x_max]
    plot_alphas = [alpha for alpha, _ in plotted]
    plot_p_zero = [value for _, value in plotted]
    xlim = (-0.5, args.x_max + 0.5)
    ylim = (args.y_min, args.y_max)
    formula_x = list(range(0, args.x_max + 1))
    formula_y = [
        0.25 - (alpha // 2 - (1 if alpha % 2 else 0)) / (2.0 * args.n)
        for alpha in formula_x
    ]
    sigma_y = [
        ((args.n / 2.0 - ((alpha + 1) // 2)) ** 0.5) / (2.0 * args.n)
        for alpha in formula_x
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if args.render == "density":
        density_cmap = LinearSegmentedColormap.from_list(
            "white_to_sky_blue",
            ["#ffffff", "#e7f7ff", "#bfefff", "#7ed3ff", "#2aa7df"],
        )
        density = gaussian_splat_density(
            plot_alphas,
            plot_p_zero,
            xlim,
            ylim,
            args.grid_x,
            args.grid_y,
            args.splat_x,
            args.splat_y,
        )
        density = np.log1p(density)
        image = ax.imshow(
            density,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            aspect="auto",
            cmap=density_cmap,
            interpolation="bilinear",
        )
        cbar = fig.colorbar(image, ax=ax, pad=0.012)
        cbar.set_label("log density")
    else:
        ax.scatter(
            plot_alphas,
            plot_p_zero,
            s=18,
            color="#2f6f9f",
            alpha=0.28,
            edgecolors="none",
            label="samples",
        )
    ax.plot(
        formula_x,
        formula_y,
        color="#c43c39",
        linestyle="--",
        linewidth=1.8,
        label=r"$\frac{1}{4}-\frac{\lfloor\alpha/2\rfloor-\mathbf{1}_{\alpha\ {\rm odd}}}{2n}$",
    )
    ax.plot(
        formula_x,
        [mean + 2.0 * sigma for mean, sigma in zip(formula_y, sigma_y)],
        color="#c43c39",
        linestyle=":",
        linewidth=1.3,
        label=r"mean $\pm\,2\sigma_\alpha$",
    )
    ax.plot(
        formula_x,
        [mean - 2.0 * sigma for mean, sigma in zip(formula_y, sigma_y)],
        color="#c43c39",
        linestyle=":",
        linewidth=1.3,
    )
    ax.set_xlabel(r"$\nu_2((q'-q)s)$")
    ax.set_ylabel("Probability of a zero cosine factor")
    if args.title:
        ax.set_title(args.title)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    if args.x_max > 50:
        tick_step = 10
    elif args.x_max > 30:
        tick_step = 5
    else:
        tick_step = 1 if args.x_max <= 16 else 2
    ax.set_xticks(list(range(0, args.x_max + 1, tick_step)))
    ax.legend(frameon=True, loc="upper right", facecolor="white", framealpha=0.9, edgecolor="none")
    ax.grid(color="#d0d0d0", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.pdf_output)
    print(f"wrote {args.output}")
    print(f"wrote {args.pdf_output}")


if __name__ == "__main__":
    main()
