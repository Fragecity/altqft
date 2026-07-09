#!/usr/bin/env python3
"""Torch check of the HP-1 dyadic-resonance scaling.

This is a dependency-light version of ``hp1_c_twisted_dyadic_cuda.py``.  It
uses torch instead of cupy, streams one dyadic-depth group at a time, and avoids
large bit masks by applying controlled phases through strided tensor views.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


DEFAULT_PAIRS = "3:5,3:7,5:7"
DEFAULT_PAIR_VALUES = "3,5,7,9,11,13,15,17,19"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHASE_BLOCK_DTYPE = torch.complex64
PROBABILITY_DTYPE = torch.float32
DEFAULT_OVERLAP_CHUNK_ELEMENTS = 16_777_216
DEFAULT_PROBABILITY_CHUNK_ELEMENTS = 16_777_216
DEFAULT_HADAMARD_CHUNK_ELEMENTS = 16_777_216
MANY_PAIR_THRESHOLD = 8


@dataclass(frozen=True)
class DyadicRow:
    n: int
    a: int
    u: int
    v: int
    s: int
    t: int
    scaled: float
    residual: float


@dataclass(frozen=True)
class RunResult:
    rows: list[DyadicRow]
    elapsed_seconds: float
    stopped_early: bool


@dataclass(frozen=True)
class WorkerResult:
    device: str
    rows: list[DyadicRow]
    elapsed_seconds: float
    stopped_early: bool


def nu2(value: int) -> int:
    return (value & -value).bit_length() - 1


def parse_pairs(raw_pairs: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in raw_pairs.split(","):
        left, right = raw_pair.split(":")
        pairs.append((int(left), int(right)))
    return pairs


def parse_pair_values(raw_values: str) -> list[int]:
    values = [int(value) for value in raw_values.split(",") if value.strip()]
    if len(values) < 2:
        raise ValueError("at least two pair values are required")
    if any(value <= 0 or value % 2 == 0 for value in values):
        raise ValueError("pair values must be positive odd integers")
    if len(set(values)) != len(values):
        raise ValueError("pair values must be unique")
    return sorted(values)


def generate_pairs(
    pair_values: list[int],
    pair_count: int,
    *,
    require_coprime: bool,
) -> list[tuple[int, int]]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")

    candidates = [
        (left, right)
        for left, right in itertools.combinations(pair_values, 2)
        if not require_coprime or math.gcd(left, right) == 1
    ]
    if pair_count > len(candidates):
        raise ValueError(
            f"requested {pair_count} pairs, but only {len(candidates)} "
            "candidate pairs are available"
        )
    return candidates[:pair_count]


def unique_pair_values(pairs: list[tuple[int, int]]) -> list[int]:
    return sorted({value for pair in pairs for value in pair})


def _phase_factor(theta: float, device: torch.device) -> Tensor:
    return torch.tensor(
        complex(math.cos(theta), math.sin(theta)),
        dtype=PHASE_BLOCK_DTYPE,
        device=device,
    )


def _apply_hadamard_slices_inplace(lower: Tensor, upper: Tensor) -> None:
    lower_original = lower.clone()
    scale = 1.0 / math.sqrt(2.0)
    lower.add_(upper).mul_(scale)
    lower_original.sub_(upper).mul_(scale)
    upper.copy_(lower_original)


def apply_hadamard_inplace(state: Tensor, qubit: int) -> None:
    stride = 1 << qubit
    block = stride << 1
    view = state.view(-1, block)
    row_count = view.shape[0]

    if stride <= DEFAULT_HADAMARD_CHUNK_ELEMENTS:
        row_chunk = max(1, DEFAULT_HADAMARD_CHUNK_ELEMENTS // stride)
        for row_start in range(0, row_count, row_chunk):
            row_stop = min(row_start + row_chunk, row_count)
            row_view = view[row_start:row_stop]
            _apply_hadamard_slices_inplace(
                row_view[:, :stride],
                row_view[:, stride:],
            )
        return

    for row_index in range(row_count):
        row_view = view[row_index : row_index + 1]
        for column_start in range(0, stride, DEFAULT_HADAMARD_CHUNK_ELEMENTS):
            column_stop = min(column_start + DEFAULT_HADAMARD_CHUNK_ELEMENTS, stride)
            _apply_hadamard_slices_inplace(
                row_view[:, column_start:column_stop],
                row_view[:, stride + column_start : stride + column_stop],
            )


def apply_controlled_phase_inplace(
    state: Tensor,
    nqubit: int,
    control: int,
    target: int,
    theta: float,
) -> None:
    selectors: list[object] = [slice(None)] * nqubit
    selectors[nqubit - control - 1] = 1
    selectors[nqubit - target - 1] = 1
    active = state.view((2,) * nqubit)[tuple(selectors)]
    active.mul_(_phase_factor(theta, state.device))


def periodic_state_probability(
    nqubit: int,
    period: int,
    device: torch.device,
    *,
    probability_device: torch.device | None = None,
) -> Tensor:
    size = 1 << nqubit
    support_count = ((size - 1) // period) + 1
    state = torch.zeros(size, dtype=PHASE_BLOCK_DTYPE, device=device)
    state[0::period] = 1.0 / math.sqrt(float(support_count))

    controls = range(0, nqubit, 2)
    targets = range(1, nqubit, 2)
    for control in controls:
        apply_hadamard_inplace(state, control)
        for target in targets:
            theta = math.pi / (2 ** abs(target - control))
            apply_controlled_phase_inplace(state, nqubit, control, target, theta)
    for target in targets:
        apply_hadamard_inplace(state, target)

    output_device = device if probability_device is None else probability_device
    probability = torch.empty(size, dtype=PROBABILITY_DTYPE, device=output_device)
    for start in range(0, size, DEFAULT_PROBABILITY_CHUNK_ELEMENTS):
        stop = min(start + DEFAULT_PROBABILITY_CHUNK_ELEMENTS, size)
        state_chunk = state[start:stop]
        probability_chunk = state_chunk.real.square()
        probability_chunk.add_(state_chunk.imag.square())
        if probability.device == state.device:
            probability[start:stop].copy_(probability_chunk)
        else:
            probability[start:stop].copy_(probability_chunk.cpu())
    return probability.to(PROBABILITY_DTYPE)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def scaled_overlap(
    nqubit: int,
    s: int,
    t: int,
    probabilities: dict[int, Tensor],
    *,
    overlap_chunk_elements: int = DEFAULT_OVERLAP_CHUNK_ELEMENTS,
    overlap_device: torch.device | None = None,
) -> tuple[float, float]:
    size = 1 << nqubit
    prob_s = probabilities[s]
    prob_t = probabilities[t]
    resolved_overlap_device = prob_t.device if overlap_device is None else overlap_device
    overlap_value = 0.0
    for start in range(0, size, overlap_chunk_elements):
        stop = min(start + overlap_chunk_elements, size)
        prob_s_chunk = prob_s[start:stop]
        if prob_s_chunk.device != resolved_overlap_device:
            prob_s_chunk = prob_s_chunk.to(resolved_overlap_device)
        prob_t_chunk = prob_t[start:stop]
        if prob_t_chunk.device != resolved_overlap_device:
            prob_t_chunk = prob_t_chunk.to(resolved_overlap_device)
        overlap_chunk = torch.sum(
            prob_s_chunk.double() * prob_t_chunk.double()
        )
        overlap_value += float(overlap_chunk.item())
    energy = size * overlap_value
    scaled = (energy - 1.0) * s * t / size
    residual = scaled / (2 ** min(nu2(s), nu2(t)))
    return scaled, residual


def build_rows(
    nqubit: int,
    max_a: int,
    pairs: list[tuple[int, int]],
    device: torch.device,
    *,
    a_values: list[int] | None = None,
    time_budget_seconds: float | None = None,
    pairwise: bool = False,
    cpu_cache_first: bool = False,
    cpu_probabilities: bool = False,
    verbose: bool = True,
) -> RunResult:
    rows: list[DyadicRow] = []
    values = unique_pair_values(pairs)
    start = time.perf_counter()
    deadline = None if time_budget_seconds is None else start + time_budget_seconds
    stopped_early = False
    group_times: list[float] = []

    if verbose:
        device_name = (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        )
        print(f"device={device_name} n={nqubit} N={1 << nqubit}")

    resolved_a_values = list(range(max_a + 1)) if a_values is None else a_values
    for a in resolved_a_values:
        group_start = time.perf_counter()
        if deadline is not None and group_times:
            projected_next = max(group_times[-3:])
            if group_start + projected_next > deadline:
                stopped_early = True
                break

        probability_device = torch.device("cpu") if cpu_probabilities else device

        if pairwise:
            for u, v in pairs:
                s = (1 << a) * u
                t = (1 << a) * v
                period_start = time.perf_counter()
                prob_s = periodic_state_probability(
                    nqubit,
                    s,
                    device,
                    probability_device=probability_device,
                )
                if cpu_cache_first and prob_s.device.type != "cpu":
                    prob_s_cpu = prob_s.cpu()
                    del prob_s
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    prob_t = periodic_state_probability(
                        nqubit,
                        t,
                        device,
                        probability_device=probability_device,
                    )
                    probabilities = {s: prob_s_cpu, t: prob_t}
                else:
                    probabilities = {
                        s: prob_s,
                        t: periodic_state_probability(
                            nqubit,
                            t,
                            device,
                            probability_device=probability_device,
                        ),
                    }
                synchronize_if_needed(device)
                scaled, residual = scaled_overlap(
                    nqubit,
                    s,
                    t,
                    probabilities,
                    overlap_device=device,
                )
                rows.append(
                    DyadicRow(
                        n=nqubit,
                        a=a,
                        u=u,
                        v=v,
                        s=s,
                        t=t,
                        scaled=scaled,
                        residual=residual,
                    )
                )
                if verbose:
                    print(
                        f"pair a={a:<2d} s={s:<6d} t={t:<6d} "
                        f"{time.perf_counter() - period_start:.4f}s"
                    )
        else:
            probabilities = {}
            for value in values:
                period = (1 << a) * value
                period_start = time.perf_counter()
                probabilities[period] = periodic_state_probability(
                    nqubit,
                    period,
                    device,
                    probability_device=probability_device,
                )
                synchronize_if_needed(device)
                if verbose:
                    print(
                        f"prob a={a:<2d} period={period:<6d} "
                        f"{time.perf_counter() - period_start:.4f}s"
                    )

            for u, v in pairs:
                s = (1 << a) * u
                t = (1 << a) * v
                scaled, residual = scaled_overlap(
                    nqubit,
                    s,
                    t,
                    probabilities,
                    overlap_device=device,
                )
                rows.append(
                    DyadicRow(
                        n=nqubit,
                        a=a,
                        u=u,
                        v=v,
                        s=s,
                        t=t,
                        scaled=scaled,
                        residual=residual,
                    )
                )
        synchronize_if_needed(device)
        group_times.append(time.perf_counter() - group_start)

    elapsed = time.perf_counter() - start
    return RunResult(rows=rows, elapsed_seconds=elapsed, stopped_early=stopped_early)


def _build_rows_worker(
    *,
    nqubit: int,
    max_a: int,
    pairs: list[tuple[int, int]],
    device_name: str,
    a_values: list[int],
    time_budget_seconds: float | None,
    pairwise: bool,
    cpu_cache_first: bool,
    cpu_probabilities: bool,
) -> WorkerResult:
    result = build_rows(
        nqubit,
        max_a,
        pairs,
        torch.device(device_name),
        a_values=a_values,
        time_budget_seconds=time_budget_seconds,
        pairwise=pairwise,
        cpu_cache_first=cpu_cache_first,
        cpu_probabilities=cpu_probabilities,
        verbose=False,
    )
    return WorkerResult(
        device=device_name,
        rows=result.rows,
        elapsed_seconds=result.elapsed_seconds,
        stopped_early=result.stopped_early,
    )


def _parse_nvidia_smi_gpu_rows() -> list[tuple[int, int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("failed to query GPUs with nvidia-smi") from error

    rows: list[tuple[int, int, int]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def idle_cuda_devices(
    *,
    max_memory_used_mib: int,
    max_utilization_percent: int,
) -> list[str]:
    devices = [
        f"cuda:{index}"
        for index, memory_used_mib, utilization_percent in _parse_nvidia_smi_gpu_rows()
        if memory_used_mib <= max_memory_used_mib
        and utilization_percent <= max_utilization_percent
    ]
    if not devices:
        raise RuntimeError(
            "no idle CUDA devices matched "
            f"memory_used_mib<={max_memory_used_mib} and "
            f"utilization_percent<={max_utilization_percent}"
        )
    return devices


def resolve_devices(
    raw_devices: str | None,
    *,
    fallback_device: str,
    idle_memory_used_mib: int,
    idle_utilization_percent: int,
) -> list[str]:
    if raw_devices is None:
        return [fallback_device]

    if raw_devices == "auto-idle":
        return idle_cuda_devices(
            max_memory_used_mib=idle_memory_used_mib,
            max_utilization_percent=idle_utilization_percent,
        )

    devices = [device.strip() for device in raw_devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must not be empty")
    return devices


def distribute_a_values(max_a: int, devices: list[str]) -> dict[str, list[int]]:
    chunks = {device: [] for device in devices}
    for index, a_value in enumerate(range(max_a + 1)):
        chunks[devices[index % len(devices)]].append(a_value)
    return {device: values for device, values in chunks.items() if values}


def build_rows_parallel(
    nqubit: int,
    max_a: int,
    pairs: list[tuple[int, int]],
    devices: list[str],
    *,
    time_budget_seconds: float | None = None,
    pairwise: bool = False,
    cpu_cache_first: bool = False,
    cpu_probabilities: bool = False,
    verbose: bool = True,
) -> RunResult:
    if len(devices) == 1:
        return build_rows(
            nqubit,
            max_a,
            pairs,
            torch.device(devices[0]),
            time_budget_seconds=time_budget_seconds,
            pairwise=pairwise,
            cpu_cache_first=cpu_cache_first,
            cpu_probabilities=cpu_probabilities,
            verbose=verbose,
        )

    start = time.perf_counter()
    a_chunks = distribute_a_values(max_a, devices)
    worker_results: list[WorkerResult] = []
    if verbose:
        print(
            "parallel devices="
            + ",".join(a_chunks)
            + " a_chunks="
            + ",".join(f"{device}:{values}" for device, values in a_chunks.items())
        )

    with ProcessPoolExecutor(
        max_workers=len(a_chunks),
        mp_context=get_context("spawn"),
    ) as executor:
        future_map = {
            executor.submit(
                _build_rows_worker,
                nqubit=nqubit,
                max_a=max_a,
                pairs=pairs,
                device_name=device,
                a_values=a_values,
                time_budget_seconds=time_budget_seconds,
                pairwise=pairwise,
                cpu_cache_first=cpu_cache_first,
                cpu_probabilities=cpu_probabilities,
            ): device
            for device, a_values in a_chunks.items()
        }
        for future in as_completed(future_map):
            worker_result = future.result()
            worker_results.append(worker_result)
            if verbose:
                print(
                    f"worker device={worker_result.device} "
                    f"rows={len(worker_result.rows)} "
                    f"seconds={worker_result.elapsed_seconds:.3f} "
                    f"stopped_early={worker_result.stopped_early}"
                )

    rows = [
        row
        for worker_result in worker_results
        for row in worker_result.rows
    ]
    rows.sort(key=lambda row: (row.a, row.u, row.v))
    return RunResult(
        rows=rows,
        elapsed_seconds=time.perf_counter() - start,
        stopped_early=any(worker_result.stopped_early for worker_result in worker_results),
    )


def print_rows(rows: list[DyadicRow], pairs: list[tuple[int, int]]) -> None:
    for u, v in pairs:
        print(f"({u}, {v})")
        for row in rows:
            if row.u == u and row.v == v:
                print(f"{row.a} {row.scaled:.5f} {row.residual:.5f}")


def write_csv(path: Path, rows: list[DyadicRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("n", "a", "u", "v", "s", "t", "scaled", "residual"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "n": row.n,
                    "a": row.a,
                    "u": row.u,
                    "v": row.v,
                    "s": row.s,
                    "t": row.t,
                    "scaled": row.scaled,
                    "residual": row.residual,
                }
            )


def plot_rows(path: Path, rows: list[DyadicRow], pairs: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        (3, 5): "#1f77b4",
        (3, 7): "#d62728",
        (5, 7): "#2ca02c",
    }
    markers = {
        (3, 5): "o",
        (3, 7): "s",
        (5, 7): "^",
    }
    max_a = max(row.a for row in rows)
    positive_scaled = [row.scaled for row in rows if row.scaled > 0]
    positive_residual = [row.residual for row in rows if row.residual > 0]
    scaled_ymax = max(70.0, max(positive_scaled, default=1.0) * 1.15)
    residual_ymin = max(0.0, min(positive_residual, default=0.55) * 0.85)
    residual_ymax = max(1.12, max(positive_residual, default=1.0) * 1.15)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    many_pairs = len(pairs) > MANY_PAIR_THRESHOLD
    pair_alpha = 0.28 if many_pairs else 1.0
    pair_linewidth = 0.8 if many_pairs else 1.5
    pair_marker = None if many_pairs else "o"

    for pair in pairs:
        pair_rows = [row for row in rows if (row.u, row.v) == pair]
        axes[0].plot(
            [row.a for row in pair_rows],
            [row.scaled for row in pair_rows],
            marker=markers.get(pair, pair_marker),
            color=colors.get(pair, "#1f77b4" if many_pairs else None),
            alpha=pair_alpha,
            linewidth=pair_linewidth,
            label=None if many_pairs else f"{pair}",
        )
    if many_pairs:
        median_scaled = [
            statistics.median(row.scaled for row in rows if row.a == a_value)
            for a_value in range(max_a + 1)
        ]
        axes[0].plot(
            list(range(max_a + 1)),
            median_scaled,
            color="black",
            linewidth=1.6,
            label=f"median ({len(pairs)} pairs)",
        )
    axes[0].plot(
        list(range(max_a + 1)),
        [2**a for a in range(max_a + 1)],
        "--",
        color="0.25",
        linewidth=1.2,
    )
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylim(0.8, scaled_ymax)
    axes[0].set_xticks(range(max_a + 1))
    axes[0].set_yticks([2**value for value in range(max_a + 1)])
    axes[0].set_xlabel("a")
    axes[0].set_ylabel(r"$(\mathcal{E}_C(s,t)-1)st/N$")
    axes[0].grid(True, color="0.85")
    axes[0].legend(frameon=False, fontsize=7)

    for pair in pairs:
        pair_rows = [row for row in rows if (row.u, row.v) == pair]
        axes[1].plot(
            [row.a for row in pair_rows],
            [row.residual for row in pair_rows],
            marker=markers.get(pair, pair_marker),
            color=colors.get(pair, "#1f77b4" if many_pairs else None),
            alpha=pair_alpha,
            linewidth=pair_linewidth,
        )
    if many_pairs:
        median_residual = [
            statistics.median(row.residual for row in rows if row.a == a_value)
            for a_value in range(max_a + 1)
        ]
        axes[1].plot(
            list(range(max_a + 1)),
            median_residual,
            color="black",
            linewidth=1.6,
        )
    axes[1].axhspan(0.65, 1.05, color="0.9", zorder=0)
    axes[1].axhline(1, color="0.25", linestyle="--", linewidth=1.2)
    axes[1].set_ylim(residual_ymin, residual_ymax)
    axes[1].set_xticks(range(max_a + 1))
    residual_tick_start = math.floor(residual_ymin * 5) / 5
    residual_tick_stop = math.ceil(residual_ymax * 5) / 5
    axes[1].set_yticks(
        [
            round(residual_tick_start + 0.2 * index, 1)
            for index in range(
                int(round((residual_tick_stop - residual_tick_start) / 0.2)) + 1
            )
        ]
    )
    axes[1].set_xlabel("a")
    axes[1].set_ylabel(r"$(\mathcal{E}_C(s,t)-1)st/(N2^a)$")
    axes[1].grid(True, color="0.85")

    fig.savefig(path)
    plt.close(fig)


def parse_benchmark_ns(raw_values: str | None) -> list[int]:
    if raw_values is None:
        return []
    return [int(value) for value in raw_values.split(",") if value]


def run_benchmarks(
    n_values: list[int],
    max_a: int,
    pairs: list[tuple[int, int]],
    devices: list[str],
    *,
    time_budget_seconds: float | None,
    pairwise: bool,
    cpu_cache_first: bool,
    cpu_probabilities: bool,
) -> None:
    total_start = time.perf_counter()
    results: list[tuple[int, float, bool]] = []
    for nqubit in n_values:
        if time_budget_seconds is not None and len(results) >= 2:
            elapsed = time.perf_counter() - total_start
            xs = torch.tensor([item[0] for item in results], dtype=torch.float64)
            ys = torch.log(torch.tensor([item[1] for item in results], dtype=torch.float64))
            design = torch.stack([xs, torch.ones_like(xs)], dim=1)
            coeffs = torch.linalg.lstsq(design, ys).solution
            predicted = float(torch.exp(coeffs[0] * nqubit + coeffs[1]).item())
            if elapsed + predicted > time_budget_seconds:
                print(
                    f"benchmark_stop n={nqubit} predicted_seconds={predicted:.3f} "
                    f"elapsed_seconds={elapsed:.3f}"
                )
                break

        result = build_rows_parallel(
            nqubit,
            max_a,
            pairs,
            devices,
            time_budget_seconds=None,
            pairwise=pairwise,
            cpu_cache_first=cpu_cache_first,
            cpu_probabilities=cpu_probabilities,
            verbose=False,
        )
        results.append((nqubit, result.elapsed_seconds, result.stopped_early))
        print(
            f"benchmark n={nqubit} seconds={result.elapsed_seconds:.6f} "
            f"stopped_early={result.stopped_early}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-a", type=int, default=6)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument(
        "--pair-count",
        type=int,
        help="Generate this many (u,v) pairs from --pair-values.",
    )
    parser.add_argument(
        "--pair-values",
        default=DEFAULT_PAIR_VALUES,
        help="Comma-separated positive odd values used with --pair-count.",
    )
    parser.add_argument(
        "--allow-noncoprime-pairs",
        action="store_true",
        help="Allow generated pairs with gcd(u,v)>1.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated devices for parallel a-sweep execution, e.g. "
            "'cuda:6,cuda:7'. Use 'auto-idle' to select idle GPUs with nvidia-smi."
        ),
    )
    parser.add_argument(
        "--idle-memory-used-mib",
        type=int,
        default=1024,
        help="Maximum used GPU memory for --devices auto-idle.",
    )
    parser.add_argument(
        "--idle-utilization-percent",
        type=int,
        default=5,
        help="Maximum GPU utilization percent for --devices auto-idle.",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Recompute each period pair independently to reduce peak memory.",
    )
    parser.add_argument(
        "--cpu-cache-first",
        action="store_true",
        help="With --pairwise, move the first probability vector in each pair to CPU before computing the second.",
    )
    parser.add_argument(
        "--cpu-probabilities",
        action="store_true",
        help="Write probability vectors to CPU memory in chunks; useful beyond the GPU probability-memory limit.",
    )
    parser.add_argument("--benchmark-n", help="Comma-separated n values to benchmark.")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        help="Stop before starting a group or benchmark whose estimated time exceeds this budget.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = (
        generate_pairs(
            parse_pair_values(args.pair_values),
            args.pair_count,
            require_coprime=not args.allow_noncoprime_pairs,
        )
        if args.pair_count is not None
        else parse_pairs(args.pairs)
    )
    devices = resolve_devices(
        args.devices,
        fallback_device=args.device,
        idle_memory_used_mib=args.idle_memory_used_mib,
        idle_utilization_percent=args.idle_utilization_percent,
    )
    benchmark_ns = parse_benchmark_ns(args.benchmark_n)

    if benchmark_ns:
        run_benchmarks(
            benchmark_ns,
            args.max_a,
            pairs,
            devices,
            time_budget_seconds=args.time_budget_seconds,
            pairwise=args.pairwise,
            cpu_cache_first=args.cpu_cache_first,
            cpu_probabilities=args.cpu_probabilities,
        )
        return

    result = build_rows_parallel(
        args.n,
        args.max_a,
        pairs,
        devices,
        time_budget_seconds=args.time_budget_seconds,
        pairwise=args.pairwise,
        cpu_cache_first=args.cpu_cache_first,
        cpu_probabilities=args.cpu_probabilities,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print_rows(result.rows, pairs)
    else:
        print(f"pair_count={len(pairs)} row_count={len(result.rows)}")
    print(
        f"elapsed_seconds={result.elapsed_seconds:.6f} "
        f"stopped_early={result.stopped_early}"
    )
    if args.csv is not None:
        write_csv(args.csv, result.rows)
        print(f"csv={args.csv}")
    if args.plot is not None:
        plot_rows(args.plot, result.rows, pairs)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
