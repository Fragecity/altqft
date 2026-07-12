#!/usr/bin/env python3
"""Exact state-vector HP-1 Chernoff window scan.

This reproduces the finite-n calculation used to check whether the
Theorem 3 lower bound C = Omega(1/r^2) is numerically tight.  The simulator
uses the Appendix E fixed-phase convention by default:

    theta_ij = pi / 2^|i-j|, i in Lambda_1, j in Lambda_2.

Qubit 0 is the most-significant bit.  This matches the direct formula
P_r(x) = |sum_q exp(pi i x^T C b_{qr})|^2 / (2^n R_r), but computes it by
full complex128 state-vector evolution.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PairMetrics:
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


@dataclass(frozen=True)
class SummaryRow:
    n: int
    period_min: int
    period_max: int
    mode: str
    r_at_minimum: int
    chernoff_lower: float
    chernoff_upper: float
    chernoff_at_candidate: float
    alpha_star_at_candidate: float
    chernoff_half_at_candidate: float
    hellinger_squared_at_candidate: float
    l2_bound_at_candidate: float
    inverse_r_squared: float
    lower_over_inverse_r_squared: float
    upper_over_inverse_r_squared: float
    probability_seconds: float
    total_seconds: float


def apply_hadamard_inplace(state: ComplexArray, qubit: int, nqubit: int) -> None:
    """Apply H to one qubit in-place; qubit 0 is the most-significant bit."""
    stride = 1 << (nqubit - 1 - qubit)
    blocks = state.reshape(-1, 2 * stride)
    lower = blocks[:, :stride].copy()
    upper = blocks[:, stride:].copy()
    scale = 1.0 / math.sqrt(2.0)
    blocks[:, :stride] = (lower + upper) * scale
    blocks[:, stride:] = (lower - upper) * scale


class HP1FixedSimulator:
    def __init__(self, nqubit: int, *, convention: str) -> None:
        self.nqubit = nqubit
        self.size = 1 << nqubit
        self.lambda_1 = tuple(range(0, nqubit, 2))
        self.lambda_2 = tuple(range(1, nqubit, 2))
        self.phase = self._build_phase(convention)

    def _build_phase(self, convention: str) -> ComplexArray:
        if convention == "appendix":
            base = math.pi
        elif convention == "main":
            base = 2.0 * math.pi
        else:
            raise ValueError(f"unknown convention: {convention}")

        indices = np.arange(self.size, dtype=np.uint64)
        bits = [
            ((indices >> (self.nqubit - 1 - qubit)) & 1).astype(np.uint8)
            for qubit in range(self.nqubit)
        ]
        phase_argument = np.zeros(self.size, dtype=np.float64)
        for control in self.lambda_1:
            control_bits = bits[control]
            for target in self.lambda_2:
                phase_argument += (
                    base / (2 ** abs(target - control))
                ) * (control_bits & bits[target])
        return np.asarray(np.exp(1j * phase_argument), dtype=np.complex128)

    def probability(self, period: int) -> FloatArray:
        support_count = ((self.size - 1) // period) + 1
        state = np.zeros(self.size, dtype=np.complex128)
        state[np.arange(support_count) * period] = 1.0 / math.sqrt(support_count)

        for qubit in self.lambda_1:
            apply_hadamard_inplace(state, qubit, self.nqubit)
        state *= self.phase
        for qubit in self.lambda_2:
            apply_hadamard_inplace(state, qubit, self.nqubit)

        return np.square(state.real) + np.square(state.imag)


def log_prob(probability: FloatArray) -> FloatArray:
    logs = np.full_like(probability, -np.inf, dtype=np.float64)
    positive = probability > 0.0
    logs[positive] = np.log(probability[positive])
    return logs


def logsumexp(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -math.inf
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.exp(finite - maximum).sum()))


def log_power_mix(log_p: FloatArray, log_q: FloatArray, alpha: float) -> float:
    with np.errstate(invalid="ignore"):
        mixed = alpha * log_p + (1.0 - alpha) * log_q
    return logsumexp(mixed)


def chernoff_information(
    probability_p: FloatArray,
    probability_q: FloatArray,
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


def alpha_half_metrics(probability_p: FloatArray, probability_q: FloatArray) -> tuple[float, float]:
    log_coeff = log_power_mix(log_prob(probability_p), log_prob(probability_q), 0.5)
    return -log_coeff, math.exp(log_coeff)


def pair_metrics(
    nqubit: int,
    period: int,
    probability_p: FloatArray,
    probability_q: FloatArray,
    *,
    exact_chernoff: bool,
    iterations: int,
) -> PairMetrics:
    started = time.perf_counter()
    chernoff_half, bhattacharyya = alpha_half_metrics(probability_p, probability_q)
    chernoff: float | None = None
    alpha_star: float | None = None
    chernoff_coeff: float | None = None
    if exact_chernoff:
        chernoff, alpha_star, chernoff_coeff = chernoff_information(
            probability_p,
            probability_q,
            iterations=iterations,
        )
    difference = probability_p - probability_q
    l2_squared = float(np.dot(difference, difference))
    tv = 0.5 * float(np.abs(difference).sum())
    return PairMetrics(
        n=nqubit,
        r=period,
        r_next=period + 1,
        chernoff=chernoff,
        alpha_star=alpha_star,
        chernoff_coeff=chernoff_coeff,
        chernoff_half=chernoff_half,
        bhattacharyya_coeff=bhattacharyya,
        hellinger_squared=max(0.0, 1.0 - min(1.0, bhattacharyya)),
        l2_squared=l2_squared,
        l2_bound=l2_squared / 8.0,
        tv=tv,
        inverse_r_squared=1.0 / float(period * period),
        seconds=time.perf_counter() - started,
    )


def scan_window(
    nqubit: int,
    *,
    period_min: int,
    period_max: int,
    mode: str,
    convention: str,
    iterations: int,
) -> tuple[list[PairMetrics], SummaryRow]:
    started = time.perf_counter()
    simulator = HP1FixedSimulator(nqubit, convention=convention)
    probability_seconds = 0.0
    rows: list[PairMetrics] = []

    probability_start = time.perf_counter()
    current_probability = simulator.probability(period_min)
    probability_seconds += time.perf_counter() - probability_start

    for period in range(period_min, period_max + 1):
        probability_start = time.perf_counter()
        next_probability = simulator.probability(period + 1)
        probability_seconds += time.perf_counter() - probability_start
        row = pair_metrics(
            nqubit,
            period,
            current_probability,
            next_probability,
            exact_chernoff=(mode == "exact"),
            iterations=iterations,
        )
        rows.append(row)
        current_probability = next_probability

    if mode == "exact":
        candidate = min(rows, key=lambda row: require_float(row.chernoff, "chernoff"))
    elif mode == "half-bound":
        candidate_index = min(range(len(rows)), key=lambda index: rows[index].chernoff_half)
        candidate = rows[candidate_index]
        probability_start = time.perf_counter()
        probability_p = simulator.probability(candidate.r)
        probability_q = simulator.probability(candidate.r_next)
        probability_seconds += time.perf_counter() - probability_start
        candidate = pair_metrics(
            nqubit,
            candidate.r,
            probability_p,
            probability_q,
            exact_chernoff=True,
            iterations=iterations,
        )
        rows[candidate_index] = candidate
    else:
        raise ValueError(f"unknown mode: {mode}")

    chernoff_value = require_float(candidate.chernoff, "chernoff")
    lower = chernoff_value if mode == "exact" else min(row.chernoff_half for row in rows)
    upper = chernoff_value
    summary = SummaryRow(
        n=nqubit,
        period_min=period_min,
        period_max=period_max,
        mode=mode,
        r_at_minimum=candidate.r,
        chernoff_lower=lower,
        chernoff_upper=upper,
        chernoff_at_candidate=chernoff_value,
        alpha_star_at_candidate=require_float(candidate.alpha_star, "alpha_star"),
        chernoff_half_at_candidate=candidate.chernoff_half,
        hellinger_squared_at_candidate=candidate.hellinger_squared,
        l2_bound_at_candidate=candidate.l2_bound,
        inverse_r_squared=candidate.inverse_r_squared,
        lower_over_inverse_r_squared=lower / candidate.inverse_r_squared,
        upper_over_inverse_r_squared=upper / candidate.inverse_r_squared,
        probability_seconds=probability_seconds,
        total_seconds=time.perf_counter() - started,
    )
    return rows, summary


def parse_n_values(raw_value: str) -> list[int]:
    if ":" in raw_value:
        parts = [int(value) for value in raw_value.split(":") if value]
        if len(parts) not in (2, 3):
            raise ValueError("range syntax is start:stop[:step]")
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        return list(range(start, stop + 1, step))
    values = [int(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one n value")
    return values


def resolve_period_max(nqubit: int, raw_period_max: int | None) -> int:
    if raw_period_max is not None:
        return raw_period_max
    return int(math.floor(math.exp2(nqubit / 2.0)))


def resolve_mode(nqubit: int, raw_mode: str, exact_up_to: int) -> str:
    if raw_mode != "auto":
        return raw_mode
    return "exact" if nqubit <= exact_up_to else "half-bound"


def require_float(value: float | None, field_name: str) -> float:
    if value is None:
        raise ValueError(f"missing required value: {field_name}")
    return value


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(rows[0])]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def print_summary(row: SummaryRow) -> None:
    if row.mode == "exact":
        value = f"{row.chernoff_at_candidate:.12g}"
    else:
        value = f"[{row.chernoff_lower:.12g}, {row.chernoff_upper:.12g}]"
    print(
        "n={n} r*={r} mode={mode} C={value} 1/r^2={inv:.6g} "
        "C/(1/r^2)=[{lo:.6g},{hi:.6g}] time={seconds:.2f}s".format(
            n=row.n,
            r=row.r_at_minimum,
            mode=row.mode,
            value=value,
            inv=row.inverse_r_squared,
            lo=row.lower_over_inverse_r_squared,
            hi=row.upper_over_inverse_r_squared,
            seconds=row.total_seconds,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="8,10,12,14,16,18")
    parser.add_argument("--period-min", type=int, default=2)
    parser.add_argument("--period-max", type=int)
    parser.add_argument(
        "--mode",
        choices=("auto", "exact", "half-bound"),
        default="auto",
        help="auto uses exact through --exact-up-to and half-bound above it.",
    )
    parser.add_argument("--exact-up-to", type=int, default=16)
    parser.add_argument("--convention", choices=("appendix", "main"), default="appendix")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hp1_chernoff/repro_exact_window"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows: list[SummaryRow] = []
    for nqubit in parse_n_values(args.n_values):
        period_max = resolve_period_max(nqubit, args.period_max)
        mode = resolve_mode(nqubit, args.mode, args.exact_up_to)
        rows, summary = scan_window(
            nqubit,
            period_min=args.period_min,
            period_max=period_max,
            mode=mode,
            convention=args.convention,
            iterations=args.iterations,
        )
        summary_rows.append(summary)
        write_csv(
            args.output_dir / f"hp1_{args.convention}_n{nqubit}_{mode}.csv",
            rows,
        )
        print_summary(summary)

    write_csv(args.output_dir / f"hp1_{args.convention}_summary.csv", summary_rows)
    print(f"summary={args.output_dir / f'hp1_{args.convention}_summary.csv'}")


if __name__ == "__main__":
    main()
