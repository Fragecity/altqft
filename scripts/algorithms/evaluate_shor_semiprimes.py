from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

from altqft.algorithms import ShorConfig, run_shor

DEFAULT_START = 11
DEFAULT_STOP = 120
DEFAULT_COUNTING_QUBITS = 11
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 7


@dataclass(frozen=True, slots=True)
class SemiprimeCase:
    N: int
    prime_factors: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    N: int
    expected_factors: tuple[int, int]
    selected_a: int | None
    selection_mode: str
    success: bool
    found_factors: tuple[int, int] | None
    order: int | None
    top_status: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate semiprimes in a range, pick a classically suitable base a for each N, "
            "and test whether the Shor demo recovers the prime factors."
        ),
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="Inclusive lower bound.")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="Inclusive upper bound.")
    parser.add_argument(
        "--counting-qubits",
        type=int,
        default=DEFAULT_COUNTING_QUBITS,
        help="Counting-register qubits used for every test.",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per Shor run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed.")
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


def semiprime_cases(start: int, stop: int) -> list[SemiprimeCase]:
    cases: list[SemiprimeCase] = []
    for N in range(start, stop + 1):
        factors = prime_factors_with_multiplicity(N)
        if len(factors) == 2:
            cases.append(SemiprimeCase(N=N, prime_factors=tuple(sorted((factors[0], factors[1])))))
    return cases


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


def choose_coprime_order_finding_a(case: SemiprimeCase) -> int | None:
    for a in range(2, case.N):
        order = multiplicative_order(a, case.N)
        if order is None:
            continue
        factors = factors_from_order(a, case.N, order)
        if factors == case.prime_factors:
            return a
    return None


def choose_gcd_shortcut_a(case: SemiprimeCase) -> int | None:
    for a in range(2, case.N):
        gcd_value = math.gcd(a, case.N)
        if 1 < gcd_value < case.N:
            factors = tuple(sorted((gcd_value, case.N // gcd_value)))
            if factors == case.prime_factors:
                return a
    return None


def evaluate_case(
    case: SemiprimeCase,
    *,
    counting_qubits: int,
    shots: int,
    seed: int,
) -> EvaluationRow:
    selected_a = choose_coprime_order_finding_a(case)
    selection_mode = "order-finding"
    if selected_a is None:
        selected_a = choose_gcd_shortcut_a(case)
        selection_mode = "gcd-shortcut"
    if selected_a is None:
        return EvaluationRow(
            N=case.N,
            expected_factors=case.prime_factors,
            selected_a=None,
            selection_mode="none",
            success=False,
            found_factors=None,
            order=None,
            top_status="no suitable a found",
        )

    result = run_shor(
        ShorConfig(
            N=case.N,
            a=selected_a,
            counting_qubits=counting_qubits,
            shots=shots,
            seed=seed,
        )
    )
    top_status = result.candidates[0].status if result.candidates else "gcd shortcut"
    return EvaluationRow(
        N=case.N,
        expected_factors=case.prime_factors,
        selected_a=selected_a,
        selection_mode=selection_mode,
        success=result.success and result.factors == case.prime_factors,
        found_factors=result.factors,
        order=result.order,
        top_status=top_status,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = semiprime_cases(int(args.start), int(args.stop))
    rows = [
        evaluate_case(
            case,
            counting_qubits=int(args.counting_qubits),
            shots=int(args.shots),
            seed=int(args.seed),
        )
        for case in cases
    ]

    print(
        "config "
        f"start={args.start} "
        f"stop={args.stop} "
        f"counting_qubits={args.counting_qubits} "
        f"shots={args.shots} "
        f"seed={args.seed}"
    )
    print(
        "summary "
        f"semiprimes={len(rows)} "
        f"successes={sum(1 for row in rows if row.success)} "
        f"failures={sum(1 for row in rows if not row.success)}"
    )
    for row in rows:
        print(
            "case "
            f"N={row.N} "
            f"expected={row.expected_factors[0]}x{row.expected_factors[1]} "
            f"a={row.selected_a} "
            f"mode={row.selection_mode} "
            f"success={row.success} "
            f"found={row.found_factors} "
            f"order={row.order} "
            f"status={row.top_status}"
        )

    return 0 if all(row.success for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
