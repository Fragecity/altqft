from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from altqft.algorithms.shor_ph1 import PH1ShorConfig, run_shor_with_ph1

DEFAULT_START = 11
DEFAULT_STOP = 120
DEFAULT_NQUBIT = 11
DEFAULT_MEASUREMENT_COUNT = 32_768
DEFAULT_TOP_K = 10
DEFAULT_SEED = 7


@dataclass(frozen=True, slots=True)
class SemiprimeCase:
    N: int
    prime_factors: tuple[int, int]
    a: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the PH1 + DeepSet replacement for QFT/continued fractions on the 19 "
            "semiprimes in 11..120 that admit a coprime order-finding base a."
        ),
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="Inclusive lower bound.")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="Inclusive upper bound.")
    parser.add_argument("--nqubit", type=int, default=DEFAULT_NQUBIT, help="PH1/DeepSet qubit count.")
    parser.add_argument(
        "--period-min",
        type=int,
        default=None,
        help="Smallest candidate period expected from the DeepSet model. Defaults to nqubit when retraining is disabled.",
    )
    parser.add_argument(
        "--period-max",
        type=int,
        default=None,
        help="Largest candidate period expected from the DeepSet model.",
    )
    parser.add_argument(
        "--measurement-count",
        type=int,
        default=DEFAULT_MEASUREMENT_COUNT,
        help="PH1 computational-basis samples per case.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many model-ranked periods to test.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared RNG seed.")
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Model directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--allow-phase-retraining",
        action="store_true",
        help="Allow PH1 phase retraining if the expected artifact is missing.",
    )
    return parser.parse_args(argv)


def prime_factors_with_multiplicity(n: int) -> list[int]:
    factors: list[int] = []
    remainder = n
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor += 1 if divisor == 2 else 2
    if remainder > 1:
        factors.append(remainder)
    return factors


def multiplicative_order(a: int, modulus: int) -> int | None:
    if math.gcd(a, modulus) != 1:
        return None
    value = 1
    for order in range(1, modulus + 1):
        value = (value * a) % modulus
        if value == 1:
            return order
    return None


def factors_from_order(a: int, modulus: int, order: int) -> tuple[int, int] | None:
    if order % 2 != 0:
        return None
    half_power = pow(a, order // 2, modulus)
    if half_power in (1, modulus - 1):
        return None
    left = math.gcd(half_power - 1, modulus)
    right = math.gcd(half_power + 1, modulus)
    if 1 < left < modulus and 1 < right < modulus:
        return tuple(sorted((left, right)))
    return None


def choose_coprime_order_finding_a(N: int, expected_factors: tuple[int, int]) -> int | None:
    for a in range(2, N):
        order = multiplicative_order(a, N)
        if order is None:
            continue
        factors = factors_from_order(a, N, order)
        if factors == expected_factors:
            return a
    return None


def order_finding_semiprimes(start: int, stop: int) -> list[SemiprimeCase]:
    cases: list[SemiprimeCase] = []
    for N in range(start, stop + 1):
        factors = prime_factors_with_multiplicity(N)
        if len(factors) != 2:
            continue
        expected_factors = tuple(sorted((factors[0], factors[1])))
        a = choose_coprime_order_finding_a(N, expected_factors)
        if a is not None:
            cases.append(SemiprimeCase(N=N, prime_factors=expected_factors, a=a))
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolved_period_min = int(args.period_min) if args.period_min is not None else int(args.nqubit)
    resolved_period_max = int(args.period_max) if args.period_max is not None else None
    cases = order_finding_semiprimes(int(args.start), int(args.stop))

    print(
        "config "
        f"start={args.start} "
        f"stop={args.stop} "
        f"nqubit={args.nqubit} "
        f"period_min={resolved_period_min} "
        f"period_max={resolved_period_max} "
        f"measurement_count={args.measurement_count} "
        f"top_k={args.top_k} "
        f"seed={args.seed} "
        f"allow_phase_retraining={args.allow_phase_retraining}"
    )
    print(f"cases count={len(cases)}")

    top1_successes = 0
    topk_successes = 0
    successful_cases: list[int] = []
    failed_cases: list[int] = []
    for case in cases:
        result = run_shor_with_ph1(
            PH1ShorConfig(
                N=case.N,
                a=case.a,
                nqubit=int(args.nqubit),
                period_min=resolved_period_min,
                period_max=resolved_period_max,
                measurement_count=int(args.measurement_count),
                top_k=int(args.top_k),
                seed=int(args.seed),
                model_dir=Path(args.model_dir),
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
                allow_phase_retraining=bool(args.allow_phase_retraining),
            )
        )
        top1_correct = result.top1_period is not None and factors_from_order(case.a, case.N, result.top1_period) == case.prime_factors
        if top1_correct:
            top1_successes += 1
        if result.success and result.factors == case.prime_factors:
            topk_successes += 1
            successful_cases.append(case.N)
        else:
            failed_cases.append(case.N)

        print(
            "case "
            f"N={case.N} "
            f"expected={case.prime_factors[0]}x{case.prime_factors[1]} "
            f"a={case.a} "
            f"top1_period={result.top1_period} "
            f"selected_period={result.predicted_period} "
            f"top1_success={top1_correct} "
            f"topk_success={result.success and result.factors == case.prime_factors} "
            f"found={result.factors} "
            f"top_periods={list(result.top_periods)}"
        )

    print(
        "summary "
        f"cases={len(cases)} "
        f"top1_successes={top1_successes} "
        f"topk_successes={topk_successes}"
    )
    print(f"successful_cases={successful_cases}")
    print(f"failed_cases={failed_cases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
