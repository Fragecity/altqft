#!/usr/bin/env python3
"""CUDA check of the HP-1 dyadic-resonance scaling.

The script simulates the fixed-phase HP-1 circuit directly on periodic input
states and reports

    ((E_C(s,t)-1) s t / N, ((E_C(s,t)-1) s t / (N 2^a))

for pairs s=2^a u, t=2^a v.  This is equivalent to evaluating
N * sum_x Pr_s(x) Pr_t(x) via the output probability vectors.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np


KERNEL_SOURCE = r"""
extern "C" __global__
void hadamard(float* re, float* im, const long long pairs, const long long stride) {
    const long long k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= pairs) {
        return;
    }

    const long long block = stride << 1;
    const long long lower = (k / stride) * block + (k % stride);
    const long long upper = lower + stride;
    const float scale = 0.70710678118654752440f;

    const float lr = re[lower];
    const float li = im[lower];
    const float ur = re[upper];
    const float ui = im[upper];

    re[lower] = (lr + ur) * scale;
    im[lower] = (li + ui) * scale;
    re[upper] = (lr - ur) * scale;
    im[upper] = (li - ui) * scale;
}

extern "C" __global__
void controlled_phase(
    float* re,
    float* im,
    const long long size,
    const int control,
    const int target,
    const float phase_re,
    const float phase_im
) {
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    if ((((idx >> control) & 1LL) == 0LL) || (((idx >> target) & 1LL) == 0LL)) {
        return;
    }

    const float old_re = re[idx];
    const float old_im = im[idx];
    re[idx] = old_re * phase_re - old_im * phase_im;
    im[idx] = old_re * phase_im + old_im * phase_re;
}

extern "C" __global__
void square_probability(
    const float* re,
    const float* im,
    float* prob,
    const long long size
) {
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    prob[idx] = re[idx] * re[idx] + im[idx] * im[idx];
}
"""


HADAMARD_KERNEL = cp.RawKernel(KERNEL_SOURCE, "hadamard")
CONTROLLED_PHASE_KERNEL = cp.RawKernel(KERNEL_SOURCE, "controlled_phase")
SQUARE_PROBABILITY_KERNEL = cp.RawKernel(KERNEL_SOURCE, "square_probability")


def nu2(value: int) -> int:
    return (value & -value).bit_length() - 1


def parse_pairs(raw_pairs: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in raw_pairs.split(","):
        left, right = raw_pair.split(":")
        pairs.append((int(left), int(right)))
    return pairs


def launch_size(count: int, block_size: int) -> tuple[tuple[int], tuple[int]]:
    return ((count + block_size - 1) // block_size,), (block_size,)


def apply_hadamard(re: cp.ndarray, im: cp.ndarray, qubit: int, block_size: int) -> None:
    pairs = int(re.size) // 2
    stride = 1 << qubit
    grid, block = launch_size(pairs, block_size)
    HADAMARD_KERNEL(grid, block, (re, im, np.int64(pairs), np.int64(stride)))


def apply_controlled_phase(
    re: cp.ndarray,
    im: cp.ndarray,
    control: int,
    target: int,
    theta: float,
    block_size: int,
) -> None:
    phase_re = math.cos(theta)
    phase_im = math.sin(theta)
    grid, block = launch_size(int(re.size), block_size)
    CONTROLLED_PHASE_KERNEL(
        grid,
        block,
        (
            re,
            im,
            np.int64(re.size),
            np.int32(control),
            np.int32(target),
            np.float32(phase_re),
            np.float32(phase_im),
        ),
    )


def hp1_probability(n: int, period: int, block_size: int) -> cp.ndarray:
    size = 1 << n
    re = cp.zeros(size, dtype=cp.float32)
    im = cp.zeros(size, dtype=cp.float32)
    support = cp.arange(0, size, period, dtype=cp.int64)
    re[support] = 1.0 / math.sqrt(int(support.size))

    controls = range(0, n, 2)
    targets = range(1, n, 2)
    for control in controls:
        apply_hadamard(re, im, control, block_size)
        for target in targets:
            theta = math.pi / (2 ** abs(target - control))
            apply_controlled_phase(re, im, control, target, theta, block_size)
    for target in targets:
        apply_hadamard(re, im, target, block_size)

    prob = cp.empty(size, dtype=cp.float32)
    grid, block = launch_size(size, block_size)
    SQUARE_PROBABILITY_KERNEL(grid, block, (re, im, prob, np.int64(size)))
    return prob


def scaled_overlap(n: int, s: int, t: int, probabilities: dict[int, cp.ndarray]) -> tuple[float, float]:
    size = 1 << n
    overlap = cp.sum(
        probabilities[s].astype(cp.float64) * probabilities[t].astype(cp.float64)
    )
    energy = size * float(overlap.get())
    scaled = (energy - 1.0) * s * t / size
    residual = scaled / (2 ** min(nu2(s), nu2(t)))
    return scaled, residual


def build_rows(n: int, max_a: int, pairs: list[tuple[int, int]], block_size: int) -> list[dict[str, float | int]]:
    periods = sorted({(1 << a) * value for a in range(max_a + 1) for pair in pairs for value in pair})
    probabilities: dict[int, cp.ndarray] = {}

    print(f"device={cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} n={n} N={1 << n}")
    print(f"periods={len(periods)} max_a={max_a}")
    start = time.perf_counter()
    for index, period in enumerate(periods, start=1):
        period_start = time.perf_counter()
        probabilities[period] = hp1_probability(n, period, block_size)
        cp.cuda.Stream.null.synchronize()
        print(
            f"prob {index:2d}/{len(periods):2d}: "
            f"s={period:<5d} R={(1 << n) // period + int((1 << n) % period != 0):<7d} "
            f"{time.perf_counter() - period_start:.3f}s"
        )

    rows: list[dict[str, float | int]] = []
    for u, v in pairs:
        for a in range(max_a + 1):
            s = (1 << a) * u
            t = (1 << a) * v
            scaled, residual = scaled_overlap(n, s, t, probabilities)
            rows.append(
                {
                    "n": n,
                    "a": a,
                    "u": u,
                    "v": v,
                    "s": s,
                    "t": t,
                    "scaled": scaled,
                    "residual": residual,
                }
            )
    cp.cuda.Stream.null.synchronize()
    print(f"elapsed={time.perf_counter() - start:.3f}s")
    return rows


def print_rows(rows: list[dict[str, float | int]], pairs: list[tuple[int, int]]) -> None:
    for u, v in pairs:
        print(f"({u}, {v})")
        for row in rows:
            if row["u"] == u and row["v"] == v:
                print(f"{row['a']} {row['scaled']:.5f} {row['residual']:.5f}")


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("n", "a", "u", "v", "s", "t", "scaled", "residual"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-a", type=int, default=6)
    parser.add_argument("--pairs", default="3:5,3:7,5:7")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    rows = build_rows(args.n, args.max_a, pairs, args.block_size)
    print_rows(rows, pairs)
    if args.csv is not None:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
