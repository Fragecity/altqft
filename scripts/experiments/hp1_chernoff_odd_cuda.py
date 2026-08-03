#!/usr/bin/env python3
"""CUDA-capable sampled-r HP-1 Chernoff curve.

This is the Torch/CUDA version of the n=16 exact curve calculation.  It keeps
the same convention as ``hp1_chernoff_window_exact.py``:

* qubit 0 is the most-significant bit,
* fixed HP-1 phases use theta_ij = pi / 2^|i-j| by default,
* state evolution uses complex128 and probability/Chernoff reductions use
  float64.

By default, r values are log-spaced as round(2^linspace(...)) and all integer
powers of two in the requested window are added explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from hp1_chernoff_mc_point_query import (
    bit_matrix_from_ints,
    importance_estimate,
    parse_prefixes,
    sample_prefix_mixture,
)


@dataclass(frozen=True)
class OddChernoffRow:
    n: int
    r: int
    r_next: int
    chernoff: float | None
    alpha_star: float | None
    chernoff_coeff: float | None
    chernoff_half: float
    bhattacharyya_coeff: float
    hellinger_squared: float
    l2_squared: float
    l2_bound: float
    tv: float
    inverse_r_squared: float
    seconds: float
    method: str = "statevector"
    sample_count: int | None = None
    chernoff_half_standard_error: float | None = None
    support_count_r: int | None = None
    support_count_next: int | None = None
    proposal: str | None = None


def apply_hadamard_inplace(state: Tensor, qubit: int, nqubit: int) -> None:
    stride = 1 << (nqubit - 1 - qubit)
    blocks = state.view(-1, 2 * stride)
    lower = blocks[:, :stride]
    upper = blocks[:, stride:]
    lower_original = lower.clone()
    upper_original = upper.clone()
    scale = 1.0 / math.sqrt(2.0)
    lower.copy_((lower_original + upper_original) * scale)
    upper.copy_((lower_original - upper_original) * scale)


class HP1CudaSimulator:
    def __init__(self, nqubit: int, *, convention: str, device: torch.device) -> None:
        self.nqubit = nqubit
        self.size = 1 << nqubit
        self.device = device
        self.lambda_1 = tuple(range(0, nqubit, 2))
        self.lambda_2 = tuple(range(1, nqubit, 2))
        self.phase = self._build_phase(convention)

    def _build_phase(self, convention: str) -> Tensor:
        if convention == "appendix":
            base = math.pi
        elif convention == "main":
            base = 2.0 * math.pi
        else:
            raise ValueError(f"unknown convention: {convention}")

        indices = torch.arange(self.size, dtype=torch.int64, device=self.device)
        bits = [
            ((indices >> (self.nqubit - 1 - qubit)) & 1).bool()
            for qubit in range(self.nqubit)
        ]
        phase_argument = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        for control in self.lambda_1:
            for target in self.lambda_2:
                phase_argument.add_(
                    (bits[control] & bits[target]).to(torch.float64),
                    alpha=base / (2 ** abs(target - control)),
                )
        return torch.exp(1j * phase_argument).to(torch.complex128)

    def probabilities(self, periods: list[int]) -> Tensor:
        state = torch.zeros(
            (len(periods), self.size),
            dtype=torch.complex128,
            device=self.device,
        )
        for row_index, period in enumerate(periods):
            support_count = ((self.size - 1) // period) + 1
            state[row_index, 0::period] = 1.0 / math.sqrt(float(support_count))

        for qubit in self.lambda_1:
            apply_hadamard_inplace(state, qubit, self.nqubit)
        state.mul_(self.phase)
        for qubit in self.lambda_2:
            apply_hadamard_inplace(state, qubit, self.nqubit)

        return state.real.square() + state.imag.square()

    def probability(self, period: int) -> Tensor:
        return self.probabilities([period])[0]


def log_prob(probability: Tensor) -> Tensor:
    logs = torch.full_like(probability, -torch.inf, dtype=torch.float64)
    positive = probability > 0.0
    logs[positive] = torch.log(probability[positive])
    return logs


def log_power_mix(log_p: Tensor, log_q: Tensor, alpha: float) -> float:
    return float(torch.logsumexp(alpha * log_p + (1.0 - alpha) * log_q, dim=0).item())


def chernoff_information(
    probability_p: Tensor,
    probability_q: Tensor,
    *,
    iterations: int,
) -> tuple[float, float, float]:
    log_p = log_prob(probability_p)
    log_q = log_prob(probability_q)
    left = 0.0
    right = 1.0
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
    x1 = left + inv_phi2 * (right - left)
    x2 = left + inv_phi * (right - left)
    f1 = log_power_mix(log_p, log_q, x1)
    f2 = log_power_mix(log_p, log_q, x2)

    for _ in range(iterations):
        if f1 < f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + inv_phi2 * (right - left)
            f1 = log_power_mix(log_p, log_q, x1)
            continue
        left = x1
        x1 = x2
        f1 = f2
        x2 = left + inv_phi * (right - left)
        f2 = log_power_mix(log_p, log_q, x2)

    alpha_star = 0.5 * (left + right)
    log_coeff = log_power_mix(log_p, log_q, alpha_star)
    return -log_coeff, alpha_star, math.exp(log_coeff)


def build_row(
    nqubit: int,
    period: int,
    probability_p: Tensor,
    probability_q: Tensor,
    *,
    iterations: int,
) -> OddChernoffRow:
    started = time.perf_counter()
    chernoff, alpha_star, coeff = chernoff_information(
        probability_p,
        probability_q,
        iterations=iterations,
    )
    half_log_coeff = log_power_mix(log_prob(probability_p), log_prob(probability_q), 0.5)
    chernoff_half = -half_log_coeff
    bhattacharyya = math.exp(half_log_coeff)
    difference = probability_p - probability_q
    l2_squared = float(torch.dot(difference, difference).item())
    tv = 0.5 * float(torch.abs(difference).sum().item())
    return OddChernoffRow(
        n=nqubit,
        r=period,
        r_next=period + 1,
        chernoff=chernoff,
        alpha_star=alpha_star,
        chernoff_coeff=coeff,
        chernoff_half=chernoff_half,
        bhattacharyya_coeff=bhattacharyya,
        hellinger_squared=max(0.0, 1.0 - min(1.0, bhattacharyya)),
        l2_squared=l2_squared,
        l2_bound=l2_squared / 8.0,
        tv=tv,
        inverse_r_squared=1.0 / float(period * period),
        seconds=time.perf_counter() - started,
    )


def build_half_rows(
    nqubit: int,
    periods: list[int],
    probability_p: Tensor,
    probability_q: Tensor,
    *,
    seconds_per_row: float,
) -> list[OddChernoffRow]:
    log_p = log_prob(probability_p)
    log_q = log_prob(probability_q)
    half_log_coeff = torch.logsumexp(0.5 * (log_p + log_q), dim=1)
    chernoff_half = -half_log_coeff
    bhattacharyya = torch.exp(half_log_coeff)
    difference = probability_p - probability_q
    l2_squared = difference.square().sum(dim=1)
    tv = 0.5 * torch.abs(difference).sum(dim=1)

    half_values = chernoff_half.detach().cpu().tolist()
    bhattacharyya_values = bhattacharyya.detach().cpu().tolist()
    l2_values = l2_squared.detach().cpu().tolist()
    tv_values = tv.detach().cpu().tolist()
    rows: list[OddChernoffRow] = []
    for index, period in enumerate(periods):
        bhattacharyya_value = float(bhattacharyya_values[index])
        l2_value = float(l2_values[index])
        rows.append(
            OddChernoffRow(
                n=nqubit,
                r=period,
                r_next=period + 1,
                chernoff=None,
                alpha_star=None,
                chernoff_coeff=None,
                chernoff_half=float(half_values[index]),
                bhattacharyya_coeff=bhattacharyya_value,
                hellinger_squared=max(0.0, 1.0 - min(1.0, bhattacharyya_value)),
                l2_squared=l2_value,
                l2_bound=l2_value / 8.0,
                tv=float(tv_values[index]),
                inverse_r_squared=1.0 / float(period * period),
                seconds=seconds_per_row,
            )
        )
    return rows


def odd_periods(period_min: int, period_max: int) -> list[int]:
    first = period_min if period_min % 2 == 1 else period_min + 1
    return list(range(first, period_max + 1, 2))


def powers_of_two(period_min: int, period_max: int) -> list[int]:
    first_exponent = math.ceil(math.log2(period_min))
    last_exponent = math.floor(math.log2(period_max))
    return [1 << exponent for exponent in range(first_exponent, last_exponent + 1)]


def logspace_periods(period_min: int, period_max: int, count: int) -> list[int]:
    if count < 2:
        raise ValueError("logspace count must be at least 2")
    if period_min < 1:
        raise ValueError("period-min must be positive")
    left = math.log2(period_min)
    right = math.log2(period_max)
    periods = {
        min(period_max, max(period_min, int(round(2.0**exponent))))
        for exponent in np.linspace(left, right, count, dtype=np.float64).tolist()
    }
    periods.update(powers_of_two(period_min, period_max))
    return sorted(periods)


def exact_logspace_periods(period_min: int, period_max: int, count: int) -> list[int]:
    if count < 2:
        raise ValueError("target period count must be at least 2")
    if period_min < 1:
        raise ValueError("period-min must be positive")
    left = math.log2(period_min)
    right = math.log2(period_max)
    required = {period_min, period_max, *powers_of_two(period_min, period_max)}
    if len(required) > count:
        raise ValueError(
            f"target period count {count} is smaller than the {len(required)} "
            "required endpoint/power-of-two periods"
        )

    candidate_count = max(8 * count, count + 1024)
    candidates = sorted(
        {
            min(period_max, max(period_min, int(round(2.0**exponent))))
            for exponent in np.linspace(left, right, candidate_count, dtype=np.float64)
        }
        | required
    )
    optional = [period for period in candidates if period not in required]
    needed = count - len(required)
    if needed > len(optional):
        raise ValueError("not enough distinct logspace periods to hit target count")

    selected = set(required)
    if needed > 0:
        centers = np.linspace(0, len(optional) - 1, needed, dtype=np.float64)
        used_optional: set[int] = set()
        for center in centers:
            base = int(round(float(center)))
            for offset in range(len(optional)):
                for candidate_index in (base - offset, base + offset):
                    if not 0 <= candidate_index < len(optional):
                        continue
                    if candidate_index in used_optional:
                        continue
                    used_optional.add(candidate_index)
                    selected.add(optional[candidate_index])
                    break
                else:
                    continue
                break
    return sorted(selected)


def resolve_periods(
    *,
    period_min: int,
    period_max: int,
    sampling: str,
    logspace_count: int,
    target_period_count: int,
) -> list[int]:
    if period_min > period_max:
        raise ValueError("period-min must not exceed period-max")
    if target_period_count:
        if sampling != "logspace":
            raise ValueError("--target-period-count is only supported for logspace sampling")
        return exact_logspace_periods(period_min, period_max, target_period_count)
    if sampling == "logspace":
        return logspace_periods(period_min, period_max, logspace_count)
    if sampling == "odd":
        return odd_periods(period_min, period_max)
    raise ValueError(f"unknown sampling mode: {sampling}")


def scan_periods(
    nqubit: int,
    *,
    periods: list[int],
    convention: str,
    device: torch.device,
    iterations: int,
    batch_size: int,
    exact_chernoff: bool,
) -> list[OddChernoffRow]:
    simulator = HP1CudaSimulator(nqubit, convention=convention, device=device)
    rows: list[OddChernoffRow] = []
    for batch_start in range(0, len(periods), batch_size):
        batch_periods = periods[batch_start : batch_start + batch_size]
        pair_periods = [
            period_value
            for period in batch_periods
            for period_value in (period, period + 1)
        ]
        started = time.perf_counter()
        probabilities = simulator.probabilities(pair_periods)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        probability_p = probabilities[0::2]
        probability_q = probabilities[1::2]
        if exact_chernoff:
            batch_rows = [
                build_row(
                    nqubit,
                    period,
                    probability_p[index],
                    probability_q[index],
                    iterations=iterations,
                )
                for index, period in enumerate(batch_periods)
            ]
        else:
            elapsed = time.perf_counter() - started
            batch_rows = build_half_rows(
                nqubit,
                batch_periods,
                probability_p,
                probability_q,
                seconds_per_row=elapsed / len(batch_periods),
            )
        rows.extend(batch_rows)
        row = min(batch_rows, key=lambda candidate: candidate.chernoff_half)
        print(
            f"{min(batch_start + batch_size, len(periods)):3d}/{len(periods):3d} "
            f"batch={len(batch_periods):<3d} best_r={row.r:<4d} "
            f"C_half={row.chernoff_half:.9f} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return rows


def scan_periods_importance(
    nqubit: int,
    *,
    periods: list[int],
    sample_count: int,
    seed: int,
    chunk_size: int,
    prefixes: tuple[int, ...],
    uniform_weight: float,
    max_support_count: int | None,
) -> list[OddChernoffRow]:
    rows: list[OddChernoffRow] = []
    for index, period in enumerate(periods, start=1):
        row = importance_estimate(
            nqubit,
            period,
            sample_count=sample_count,
            seed=seed + period * 1000003 + sample_count,
            chunk_size=chunk_size,
            prefixes=prefixes,
            uniform_weight=uniform_weight,
            max_support_count=max_support_count,
        )
        bhattacharyya = row.coefficient_estimate
        output_row = OddChernoffRow(
            n=nqubit,
            r=period,
            r_next=period + 1,
            chernoff=None,
            alpha_star=None,
            chernoff_coeff=None,
            chernoff_half=row.mc_chernoff_half,
            bhattacharyya_coeff=bhattacharyya,
            hellinger_squared=max(0.0, 1.0 - min(1.0, bhattacharyya)),
            l2_squared=math.nan,
            l2_bound=math.nan,
            tv=math.nan,
            inverse_r_squared=1.0 / float(period * period),
            seconds=row.seconds,
            method="mc-is",
            sample_count=sample_count,
            chernoff_half_standard_error=row.mc_standard_error,
            support_count_r=row.support_count_r,
            support_count_next=row.support_count_next,
            proposal=row.proposal,
        )
        rows.append(output_row)
        best_row = min(rows, key=lambda candidate: candidate.chernoff_half)
        print(
            f"{index:3d}/{len(periods):3d} r={period:<30d} "
            f"R=({row.support_count_r},{row.support_count_next}) "
            f"C_half={row.mc_chernoff_half:.9f} "
            f"se={row.mc_standard_error:.3g} "
            f"best_r={best_row.r} seconds={row.seconds:.2f}",
            flush=True,
        )
    return rows


def build_hp1_phase_matrix_torch(nqubit: int, device: torch.device) -> Tensor:
    phase = math.pi * torch.eye(nqubit, dtype=torch.float64, device=device)
    for control in range(0, nqubit, 2):
        for target in range(1, nqubit, 2):
            phase[control, target] += math.pi / (2.0 ** abs(target - control))
    return phase


def support_bits_torch(
    nqubit: int,
    period: int,
    support_count: int,
    device: torch.device,
) -> Tensor:
    support_bits = bit_matrix_from_ints(
        [q_value * period for q_value in range(support_count)],
        nqubit,
    )
    return torch.as_tensor(support_bits.T, dtype=torch.float64, device=device)


def probability_bits_torch(
    x_bits: Tensor,
    phase_matrix: Tensor,
    nqubit: int,
    period: int,
    *,
    chunk_size: int,
    max_support_count: int | None,
) -> tuple[Tensor, int]:
    support_count = (((1 << nqubit) - 1) // period) + 1
    if max_support_count is not None and support_count > max_support_count:
        raise ValueError(
            f"period {period} has support_count={support_count}, "
            f"above max_support_count={max_support_count}"
        )

    support_bits = support_bits_torch(nqubit, period, support_count, x_bits.device)
    phase_weights = phase_matrix @ support_bits
    denominator = math.ldexp(float(support_count), nqubit)
    chunk = chunk_size if chunk_size > 0 else x_bits.shape[0]
    values: list[Tensor] = []
    for start in range(0, x_bits.shape[0], chunk):
        stop = min(start + chunk, x_bits.shape[0])
        phases = x_bits[start:stop] @ phase_weights
        amplitude_real = torch.cos(phases).sum(dim=1)
        amplitude_imag = torch.sin(phases).sum(dim=1)
        values.append((amplitude_real.square() + amplitude_imag.square()) / denominator)
    return torch.cat(values), support_count


def importance_estimate_torch(
    nqubit: int,
    period: int,
    *,
    sample_count: int,
    seed: int,
    chunk_size: int,
    prefixes: tuple[int, ...],
    uniform_weight: float,
    max_support_count: int | None,
    device: torch.device,
) -> OddChernoffRow:
    started = time.perf_counter()
    x_bits_np, proposal_np, proposal = sample_prefix_mixture(
        nqubit,
        sample_count=sample_count,
        rng=np.random.default_rng(seed),
        prefixes=prefixes,
        uniform_weight=uniform_weight,
    )
    with torch.inference_mode():
        phase_matrix = build_hp1_phase_matrix_torch(nqubit, device)
        x_bits = torch.as_tensor(x_bits_np, dtype=torch.float64, device=device)
        proposal_probability = torch.as_tensor(proposal_np, dtype=torch.float64, device=device)
        probability_p, support_count_p = probability_bits_torch(
            x_bits,
            phase_matrix,
            nqubit,
            period,
            chunk_size=chunk_size,
            max_support_count=max_support_count,
        )
        probability_q, support_count_q = probability_bits_torch(
            x_bits,
            phase_matrix,
            nqubit,
            period + 1,
            chunk_size=chunk_size,
            max_support_count=max_support_count,
        )
        samples = torch.sqrt(probability_p * probability_q) / proposal_probability
        coefficient = float(samples.mean().item())
        coefficient_se = float(samples.std(unbiased=True).item() / math.sqrt(sample_count))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    chernoff_half = -math.log(max(coefficient, np.finfo(np.float64).tiny))
    chernoff_se = coefficient_se / max(coefficient, np.finfo(np.float64).tiny)
    return OddChernoffRow(
        n=nqubit,
        r=period,
        r_next=period + 1,
        chernoff=None,
        alpha_star=None,
        chernoff_coeff=None,
        chernoff_half=chernoff_half,
        bhattacharyya_coeff=coefficient,
        hellinger_squared=max(0.0, 1.0 - min(1.0, coefficient)),
        l2_squared=math.nan,
        l2_bound=math.nan,
        tv=math.nan,
        inverse_r_squared=1.0 / float(period * period),
        seconds=time.perf_counter() - started,
        method="mc-is-torch",
        sample_count=sample_count,
        chernoff_half_standard_error=chernoff_se,
        support_count_r=support_count_p,
        support_count_next=support_count_q,
        proposal=proposal,
    )


def torch_importance_worker(task: tuple) -> OddChernoffRow:
    (
        nqubit,
        period,
        sample_count,
        seed,
        chunk_size,
        prefixes,
        uniform_weight,
        max_support_count,
        device_name,
    ) = task
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return importance_estimate_torch(
        nqubit,
        period,
        sample_count=sample_count,
        seed=seed,
        chunk_size=chunk_size,
        prefixes=prefixes,
        uniform_weight=uniform_weight,
        max_support_count=max_support_count,
        device=device,
    )


def resolve_mc_devices(raw_devices: str) -> list[str]:
    if raw_devices == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return ["cpu"]
    devices = [value.strip() for value in raw_devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("expected at least one device")
    resolved: list[str] = []
    for value in devices:
        resolved.append(value if ":" in value or value == "cpu" else f"cuda:{value}")
    return resolved


def scan_periods_importance_torch(
    nqubit: int,
    *,
    periods: list[int],
    sample_count: int,
    seed: int,
    chunk_size: int,
    prefixes: tuple[int, ...],
    uniform_weight: float,
    max_support_count: int | None,
    devices: list[str],
) -> list[OddChernoffRow]:
    tasks = [
        (
            nqubit,
            period,
            sample_count,
            seed + period * 1000003 + sample_count,
            chunk_size,
            prefixes,
            uniform_weight,
            max_support_count,
            devices[index % len(devices)],
        )
        for index, period in enumerate(periods)
    ]
    rows: list[OddChernoffRow] = []
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(devices),
        mp_context=context,
    ) as executor:
        futures = {executor.submit(torch_importance_worker, task): task[1] for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            best_row = min(rows, key=lambda candidate: candidate.chernoff_half)
            print(
                f"{index:3d}/{len(periods):3d} r={row.r:<62d} "
                f"R=({row.support_count_r},{row.support_count_next}) "
                f"C_half={row.chernoff_half:.9f} "
                f"se={row.chernoff_half_standard_error:.3g} "
                f"best_r={best_row.r} seconds={row.seconds:.2f} "
                f"elapsed={time.perf_counter() - started:.1f}",
                flush=True,
            )
    return sorted(rows, key=lambda row: row.r)


def write_csv(path: Path, rows: list[OddChernoffRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(OddChernoffRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def fit_upper_envelope(rows: list[OddChernoffRow]) -> tuple[float, float] | None:
    points = [(row.r, row.chernoff_half) for row in rows if is_power_of_two(row.r)]
    if len(points) < 2:
        return None
    x_values = np.array([math.log2(period) for period, _ in points], dtype=np.float64)
    y_values = np.array([value for _, value in points], dtype=np.float64)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return float(slope), float(intercept)


def record_low_rows(rows: list[OddChernoffRow]) -> list[OddChernoffRow]:
    records: list[OddChernoffRow] = []
    best = math.inf
    for row in sorted(rows, key=lambda candidate: candidate.r):
        if row.chernoff_half < best:
            records.append(row)
            best = row.chernoff_half
    return records


def fit_tail_lower_envelope(
    rows: list[OddChernoffRow],
    *,
    cutoff: int,
) -> tuple[float, float, int] | None:
    points = [row for row in record_low_rows(rows) if row.r >= cutoff]
    if len(points) < 2:
        return None
    x_values = np.log(np.array([row.r for row in points], dtype=np.float64))
    y_values = np.log(np.array([row.chernoff_half for row in points], dtype=np.float64))
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return float(math.exp(intercept)), float(-slope), points[0].r


def compact_coefficient_tex(value: float) -> str:
    if value == 0.0 or 1.0e-2 <= abs(value) < 1.0e4:
        return f"{value:.1f}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.1f}\times10^{{{exponent}}}"


def plot_rows(
    path_stem: Path,
    rows: list[OddChernoffRow],
    *,
    show_envelope_fits: bool,
) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    positive_rows = [
        row
        for row in rows
        if math.isfinite(row.chernoff_half) and row.chernoff_half > 0.0
    ]
    if not positive_rows:
        raise ValueError("no positive C_1/2 values are available for log-log plotting")
    r_values = [row.r for row in positive_rows]
    chernoff_half = [row.chernoff_half for row in positive_rows]
    r_min = min(r_values)
    r_max = max(r_values)
    y_min = min(chernoff_half)
    y_max = max(chernoff_half)
    nqubit = positive_rows[0].n

    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.loglog(
        r_values,
        chernoff_half,
        color="#1f77b4",
        marker="o",
        markersize=2.7,
        linewidth=1.5,
        label=r"$C_{1/2}$",
    )

    if show_envelope_fits:
        r_min_float = float(r_min)
        r_max_float = float(r_max)
        dense_r = np.geomspace(r_min_float, r_max_float, 512)
        upper_fit = fit_upper_envelope(positive_rows)
        if upper_fit is not None:
            slope, intercept = upper_fit
            ax.plot(
                dense_r,
                intercept + slope * np.log2(dense_r),
                linestyle="--",
                linewidth=1.2,
                color="#ff7f0e",
                label=rf"upper fit ${slope:.2f}\log_2 r+{intercept:.2f}$",
            )

        lower_fit = fit_tail_lower_envelope(positive_rows, cutoff=70)
        if lower_fit is not None:
            amplitude, exponent, lower_start = lower_fit
            lower_r = np.geomspace(float(lower_start), r_max_float, 256)
            ax.plot(
                lower_r,
                amplitude * lower_r ** (-exponent),
                linestyle="--",
                linewidth=1.2,
                color="#d62728",
                label=rf"lower fit ${compact_coefficient_tex(amplitude)}r^{{-{exponent:.2f}}}$",
            )

    ax.set_xlabel(r"Period $r$")
    ax.set_ylabel(r"$C_{1/2}$")
    ax.set_title(rf"Fixed-phase HP-1 $C_{{1/2}}$ scan, sampled $r$, $n={nqubit}$")
    ax.set_ylim(max(y_min * 0.65, np.finfo(np.float64).tiny), y_max * 1.45)
    ax.grid(True, which="both", color="0.88", linewidth=0.65)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(path_stem.with_suffix(".png"), dpi=240)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def resolve_device(raw_device: str, *, allow_cpu_fallback: bool) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise RuntimeError("CUDA is not available")
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_cpu_fallback:
            print("CUDA is not available; falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("CUDA is not available")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--period-min", type=int, default=2)
    parser.add_argument("--period-max", type=int, default=256)
    parser.add_argument(
        "--period-min-exponent",
        type=float,
        default=None,
        help="Override period-min with ceil(2^value).",
    )
    parser.add_argument(
        "--period-max-exponent",
        type=float,
        default=None,
        help="Override period-max with floor(2^value).",
    )
    parser.add_argument(
        "--sampling",
        choices=("logspace", "odd"),
        default="logspace",
        help="r-grid: logspace adds all powers of two; odd keeps every odd r.",
    )
    parser.add_argument(
        "--logspace-count",
        type=int,
        default=100,
        help="Number of 2^linspace samples before powers-of-two are merged in.",
    )
    parser.add_argument(
        "--target-period-count",
        type=int,
        default=0,
        help="For logspace sampling, choose exactly this many r values while keeping powers of two.",
    )
    parser.add_argument("--convention", choices=("appendix", "main"), default="appendix")
    parser.add_argument(
        "--method",
        choices=("statevector", "mc-is"),
        default="statevector",
        help="statevector is exact over all x; mc-is samples x with importance weights.",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Odd-r values per GPU batch; each batch evolves 2*batch-size states.",
    )
    parser.add_argument(
        "--exact-chernoff",
        action="store_true",
        help="Also optimize alpha for exact Chernoff. Slower; plotting still uses C_1/2.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Run on CPU when CUDA is unavailable.",
    )
    parser.add_argument("--mc-sample-count", type=int, default=4096)
    parser.add_argument("--mc-seed", type=int, default=20260713)
    parser.add_argument("--mc-chunk-size", type=int, default=256)
    parser.add_argument(
        "--mc-backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="Backend for --method mc-is. torch can use CUDA and multiple devices.",
    )
    parser.add_argument(
        "--mc-devices",
        default="auto",
        help="Comma-separated CUDA device ids/names for --mc-backend torch, e.g. 6,7.",
    )
    parser.add_argument(
        "--mc-prefixes",
        default="auto",
        help="Comma-separated zero-prefix depths, 'auto', or 'none'.",
    )
    parser.add_argument("--mc-uniform-weight", type=float, default=0.5)
    parser.add_argument(
        "--max-support-count",
        type=int,
        default=0,
        help="For mc-is, abort when R_r is larger than this. 0 means no cap.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/hp1_chernoff/repro_sampled_cuda/hp1_appendix_n16_logspace.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("figs/fi_fig/hp1_chernoff_repro_n16_logspace_cuda"),
        help="Output stem; .png and .pdf are written.",
    )
    parser.add_argument(
        "--show-envelope-fits",
        action="store_true",
        help="Draw dashed upper/lower envelope fits on the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    period_min = args.period_min
    period_max = args.period_max
    if args.period_min_exponent is not None:
        period_min = max(1, int(math.ceil(2.0 ** args.period_min_exponent)))
    if args.period_max_exponent is not None:
        period_max = max(1, int(math.floor(2.0 ** args.period_max_exponent)))

    periods = resolve_periods(
        period_min=period_min,
        period_max=period_max,
        sampling=args.sampling,
        logspace_count=args.logspace_count,
        target_period_count=args.target_period_count,
    )
    if args.method == "statevector":
        device = resolve_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
        print(
            f"method=statevector device={device} n={args.n} sampling={args.sampling} "
            f"period_count={len(periods)} batch_size={args.batch_size} "
            f"exact_chernoff={args.exact_chernoff}"
        )
        rows = scan_periods(
            args.n,
            periods=periods,
            convention=args.convention,
            device=device,
            iterations=args.iterations,
            batch_size=args.batch_size,
            exact_chernoff=args.exact_chernoff,
        )
    else:
        max_support_count = args.max_support_count or None
        if args.n > 30 and max_support_count is None:
            raise ValueError("set --max-support-count for mc-is scans with n > 30")
        prefixes = parse_prefixes(args.mc_prefixes, args.n)
        uniform_weight = args.mc_uniform_weight if prefixes else 1.0
        print(
            f"method=mc-is n={args.n} sampling={args.sampling} "
            f"period_count={len(periods)} sample_count={args.mc_sample_count} "
            f"chunk_size={args.mc_chunk_size} max_support_count={max_support_count} "
            f"prefixes={prefixes} uniform_weight={uniform_weight} "
            f"backend={args.mc_backend}"
        )
        if args.mc_backend == "torch":
            devices = resolve_mc_devices(args.mc_devices)
            print(f"mc_devices={devices}")
            rows = scan_periods_importance_torch(
                args.n,
                periods=periods,
                sample_count=args.mc_sample_count,
                seed=args.mc_seed,
                chunk_size=args.mc_chunk_size,
                prefixes=prefixes,
                uniform_weight=uniform_weight,
                max_support_count=max_support_count,
                devices=devices,
            )
        else:
            rows = scan_periods_importance(
                args.n,
                periods=periods,
                sample_count=args.mc_sample_count,
                seed=args.mc_seed,
                chunk_size=args.mc_chunk_size,
                prefixes=prefixes,
                uniform_weight=uniform_weight,
                max_support_count=max_support_count,
            )
    write_csv(args.csv_output, rows)
    plot_rows(args.plot_output, rows, show_envelope_fits=args.show_envelope_fits)
    min_row = min(rows, key=lambda row: row.chernoff_half)
    print(f"csv={args.csv_output}")
    print(f"plot={args.plot_output.with_suffix('.png')}")
    print(f"minimum r={min_row.r} C_half={min_row.chernoff_half:.12g}")


if __name__ == "__main__":
    main()
