from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from altqft.algorithms.shor_ph1 import PH1ShorConfig, run_shor_with_ph1
from altqft.nn.periods import build_period_range

DEFAULT_START = 11
DEFAULT_STOP = 120
DEFAULT_NQUBIT = 11
DEFAULT_MEASUREMENT_COUNT = 32_768
DEFAULT_TOP_K = 4
DEFAULT_SEED = 7
DEFAULT_FALLBACK_A = 13


@dataclass(frozen=True, slots=True)
class SemiprimeCase:
    N: int
    prime_factors: tuple[int, int]
    a: int
    order: int | None
    used_default_a: bool = False


@dataclass(frozen=True, slots=True)
class BaseSelection:
    a: int
    order: int | None
    used_default_a: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the PH1 + DeepSet replacement for QFT/continued fractions on semiprimes "
            "in the requested range. When no period-range-compatible coprime base exists, "
            "fall back to a default base a."
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
    parser.add_argument(
        "--default-a",
        type=int,
        default=DEFAULT_FALLBACK_A,
        help="Fallback base a used when no period-range-compatible coprime base is found.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Model directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
        help="Optional JSON file with preselected semiprime cases and bases a.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path where per-case PH1+NN Shor results are written as JSON.",
    )
    parser.add_argument(
        "--variant-tag",
        type=str,
        default=None,
        help="Optional artifact variant tag appended to PH1 and DeepSet run names.",
    )
    parser.add_argument(
        "--ph1-objective",
        choices=("min_fi", "shift_ce_mean", "hp1_shared_fi_shift"),
        default="min_fi",
        help="Expected PH1 objective for the optimized phase artifact.",
    )
    parser.add_argument(
        "--ph1-ansatz",
        choices=("HP1", "HP1_shared"),
        default="HP1",
        help="Expected PH1 ansatz for the optimized phase artifact.",
    )
    parser.add_argument(
        "--ph1-model-stem",
        type=str,
        default=None,
        help="Optional PH1 model stem for historical artifact names.",
    )
    parser.add_argument(
        "--exact-support",
        action="store_true",
        help="Expect exact-support PH1 artifacts.",
    )
    parser.add_argument(
        "--allow-phase-retraining",
        action="store_true",
        help="Allow PH1 phase retraining if the expected artifact is missing.",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only enumerate semiprimes and choose the base a without running PH1 inference.",
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
        smaller, larger = sorted((left, right))
        return smaller, larger
    return None


def choose_coprime_order_finding_a(
    N: int,
    expected_factors: tuple[int, int],
    *,
    candidate_periods: Sequence[int],
    default_a: int | None,
) -> BaseSelection | None:
    allowed_periods = set(int(value) for value in candidate_periods)
    best_selection: BaseSelection | None = None
    for a in range(2, N):
        order = multiplicative_order(a, N)
        if order is None or order not in allowed_periods:
            continue
        factors = factors_from_order(a, N, order)
        if factors == expected_factors:
            candidate = BaseSelection(a=a, order=order, used_default_a=False)
            if best_selection is None or (candidate.order, candidate.a) < (best_selection.order, best_selection.a):
                best_selection = candidate
    if best_selection is not None:
        return best_selection

    if default_a is None or not 1 < default_a < N:
        return None
    return BaseSelection(
        a=default_a,
        order=multiplicative_order(default_a, N),
        used_default_a=True,
    )


def order_finding_semiprimes(
    start: int,
    stop: int,
    *,
    candidate_periods: Sequence[int],
    default_a: int | None,
) -> tuple[list[SemiprimeCase], list[int], list[int]]:
    cases: list[SemiprimeCase] = []
    uncovered_cases: list[int] = []
    default_a_cases: list[int] = []
    for N in range(start, stop + 1):
        factors = prime_factors_with_multiplicity(N)
        if len(factors) != 2:
            continue
        smaller, larger = sorted((factors[0], factors[1]))
        expected_factors = (smaller, larger)
        selection = choose_coprime_order_finding_a(
            N,
            expected_factors,
            candidate_periods=candidate_periods,
            default_a=default_a,
        )
        if selection is None:
            uncovered_cases.append(N)
            continue
        if selection.used_default_a:
            default_a_cases.append(N)
        cases.append(
            SemiprimeCase(
                N=N,
                prime_factors=expected_factors,
                a=selection.a,
                order=selection.order,
                used_default_a=selection.used_default_a,
            )
        )
    return cases, uncovered_cases, default_a_cases


def load_semiprime_cases(path: Path) -> tuple[list[SemiprimeCase], list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"case file must contain a JSON object: {path}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"case file missing cases list: {path}")

    cases: list[SemiprimeCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError(f"invalid case entry in {path}: {item!r}")
        factors = item.get("prime_factors")
        if (
            not isinstance(factors, list)
            or len(factors) != 2
            or not all(isinstance(value, int) for value in factors)
        ):
            raise ValueError(f"invalid prime_factors in {path}: {item!r}")
        order = item.get("order")
        smaller, larger = sorted((int(factors[0]), int(factors[1])))
        cases.append(
            SemiprimeCase(
                N=int(item["N"]),
                prime_factors=(smaller, larger),
                a=int(item["a"]),
                order=int(order) if order is not None else None,
                used_default_a=bool(item.get("used_default_a", False)),
            )
        )
    uncovered = payload.get("uncovered_case_ids", [])
    if not isinstance(uncovered, list) or not all(
        isinstance(value, int) for value in uncovered
    ):
        raise ValueError(f"invalid uncovered_case_ids in {path}")
    default_a_cases = payload.get("default_a_case_ids", [])
    if not isinstance(default_a_cases, list) or not all(
        isinstance(value, int) for value in default_a_cases
    ):
        raise ValueError(f"invalid default_a_case_ids in {path}")
    return cases, uncovered, default_a_cases


def max_supported_modulus(nqubit: int) -> int:
    return 1 << nqubit


def success_rank(case: SemiprimeCase, ranked_periods: Sequence[int]) -> int | None:
    for index, period in enumerate(ranked_periods, start=1):
        if factors_from_order(case.a, case.N, int(period)) == case.prime_factors:
            return index
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_start = int(args.start)
    requested_stop = int(args.stop)
    state_space_limit = max_supported_modulus(int(args.nqubit))
    effective_stop = min(requested_stop, state_space_limit)
    resolved_period_min = int(args.period_min) if args.period_min is not None else int(args.nqubit)
    resolved_period_max = int(args.period_max) if args.period_max is not None else None
    candidate_periods = tuple(
        build_period_range(
            int(args.nqubit),
            min_period=resolved_period_min,
            max_period=resolved_period_max,
        )
    )
    if args.case_file is not None:
        cases, uncovered_cases, default_a_cases = load_semiprime_cases(Path(args.case_file))
        cases = [
            case for case in cases if requested_start <= case.N <= effective_stop
        ]
        uncovered_cases = [
            case_id
            for case_id in uncovered_cases
            if requested_start <= case_id <= effective_stop
        ]
        default_a_cases = [
            case_id
            for case_id in default_a_cases
            if requested_start <= case_id <= effective_stop
        ]
        selection_source = str(args.case_file)
    else:
        cases, uncovered_cases, default_a_cases = order_finding_semiprimes(
            requested_start,
            effective_stop,
            candidate_periods=candidate_periods,
            default_a=int(args.default_a) if args.default_a is not None else None,
        )
        selection_source = "computed"

    print(
        "config "
        f"start={requested_start} "
        f"requested_stop={requested_stop} "
        f"effective_stop={effective_stop} "
        f"state_space_limit={state_space_limit} "
        f"nqubit={args.nqubit} "
        f"period_min={resolved_period_min} "
        f"period_max={resolved_period_max} "
        f"candidate_period_count={len(candidate_periods)} "
        f"measurement_count={args.measurement_count} "
        f"top_k={args.top_k} "
        f"seed={args.seed} "
        f"default_a={args.default_a} "
        f"variant_tag={args.variant_tag} "
        f"ph1_objective={args.ph1_objective} "
        f"ph1_ansatz={args.ph1_ansatz} "
        f"ph1_model_stem={args.ph1_model_stem} "
        f"exact_support={args.exact_support} "
        f"allow_phase_retraining={args.allow_phase_retraining} "
        f"selection_source={selection_source} "
        f"select_only={args.select_only}"
    )
    skipped_start = max(requested_start, state_space_limit + 1)
    if skipped_start <= requested_stop:
        print(
            "skipped "
            f"requested_N_range={skipped_start}..{requested_stop} "
            "reason=exceeds_nqubit_state_space"
        )
    print(
        "selection "
        f"cases={len(cases)} "
        f"default_a_cases={len(default_a_cases)} "
        f"uncovered_cases={len(uncovered_cases)}"
    )
    if default_a_cases:
        print(f"default_a_case_ids={default_a_cases}")
    if uncovered_cases:
        print(f"uncovered_case_ids={uncovered_cases}")

    if args.select_only:
        for case in cases:
            print(
                "case "
                f"N={case.N} "
                f"expected={case.prime_factors[0]}x{case.prime_factors[1]} "
                f"a={case.a} "
                f"used_default_a={case.used_default_a} "
                f"true_period={case.order}"
            )
        return 0

    top1_successes = 0
    topk_successes = 0
    successful_cases: list[int] = []
    failed_cases: list[int] = []
    result_rows: list[dict[str, object]] = []
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
                variant_tag=args.variant_tag,
                ph1_objective=str(args.ph1_objective),
                ph1_ansatz=str(args.ph1_ansatz),
                ph1_model_stem=args.ph1_model_stem,
                exact_support=bool(args.exact_support),
            )
        )
        found_k = success_rank(case, result.top_periods)
        top1_correct = (
            result.top1_period is not None
            and factors_from_order(case.a, case.N, result.top1_period) == case.prime_factors
        )
        topk_correct = result.success and result.factors == case.prime_factors
        if top1_correct:
            top1_successes += 1
        if topk_correct:
            topk_successes += 1
            successful_cases.append(case.N)
        else:
            failed_cases.append(case.N)

        print(
            "case "
            f"N={case.N} "
            f"expected={case.prime_factors[0]}x{case.prime_factors[1]} "
            f"a={case.a} "
            f"used_default_a={case.used_default_a} "
            f"true_period={case.order} "
            f"top1_period={result.top1_period} "
            f"selected_period={result.predicted_period} "
            f"found_k={found_k} "
            f"top1_success={top1_correct} "
            f"topk_success={topk_correct} "
            f"found={result.factors} "
            f"top_periods={list(result.top_periods)}"
        )
        result_rows.append(
            {
                "N": case.N,
                "expected_factors": list(case.prime_factors),
                "a": case.a,
                "used_default_a": case.used_default_a,
                "true_period": case.order,
                "top1_period": result.top1_period,
                "selected_period": result.predicted_period,
                "found_k": found_k,
                "top1_success": top1_correct,
                "topk_success": topk_correct,
                "found_factors": list(result.factors) if result.factors else None,
                "top_periods": list(result.top_periods),
                "top_scores": list(getattr(result, "top_scores", ())),
            }
        )

    print(
        "summary "
        f"cases={len(cases)} "
        f"default_a_cases={len(default_a_cases)} "
        f"top1_successes={top1_successes} "
        f"topk_successes={topk_successes}"
    )
    print(f"successful_cases={successful_cases}")
    print(f"failed_cases={failed_cases}")
    if args.results_json is not None:
        results_path = Path(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(
                {
                    "config": {
                        "start": requested_start,
                        "requested_stop": requested_stop,
                        "effective_stop": effective_stop,
                        "nqubit": int(args.nqubit),
                        "period_min": resolved_period_min,
                        "period_max": resolved_period_max,
                        "measurement_count": int(args.measurement_count),
                        "top_k": int(args.top_k),
                        "seed": int(args.seed),
                        "selection_source": selection_source,
                    },
                    "summary": {
                        "cases": len(cases),
                        "default_a_cases": len(default_a_cases),
                        "top1_successes": top1_successes,
                        "topk_successes": topk_successes,
                        "successful_cases": successful_cases,
                        "failed_cases": failed_cases,
                    },
                    "cases": result_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"results_json={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
