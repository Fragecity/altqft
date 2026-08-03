from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from altqft.algorithms.shor_ph1 import PH1ShorConfig, run_shor_with_ph1

DEFAULT_N = 15
DEFAULT_A = 2
DEFAULT_NQUBIT = 4
DEFAULT_MEASUREMENT_COUNT = 16_384
DEFAULT_TOP_K = 4
DEFAULT_SEED = 7


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace inverse QFT and continued fractions with optimized PH1 plus a DeepSet "
            "period-recovery network for a Shor-style factorization demo."
        ),
    )
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Composite integer to factor.")
    parser.add_argument("--a", type=int, default=DEFAULT_A, help="Base used in modular exponentiation.")
    parser.add_argument("--nqubit", type=int, default=DEFAULT_NQUBIT, help="Counting-register qubits.")
    parser.add_argument(
        "--period-min",
        type=int,
        default=2,
        help="Smallest candidate period expected from the DeepSet model.",
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
        help="PH1 computational-basis samples turned into a DeepSet bitmatrix.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many period guesses to print.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--prefer-smoke-artifacts",
        action="store_true",
        help="Prefer model/smoke artifacts over the full 4q checkpoints.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Model artifact directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PH1ShorConfig(
        N=int(args.N),
        a=int(args.a),
        nqubit=int(args.nqubit),
        period_min=int(args.period_min),
        period_max=args.period_max,
        measurement_count=int(args.measurement_count),
        top_k=int(args.top_k),
        seed=int(args.seed),
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        prefer_smoke_artifacts=bool(args.prefer_smoke_artifacts),
    )
    result = run_shor_with_ph1(config)

    print(
        "config "
        f"N={config.N} "
        f"a={config.a} "
        f"nqubit={config.nqubit} "
        f"period_range={list(config.candidate_periods)} "
        f"measurement_count={config.measurement_count} "
        f"top_k={config.top_k} "
        f"seed={config.seed}"
    )
    print(
        "periodic_state "
        f"measured_work_value={result.measured_work_value} "
        f"support_x={list(result.support_x)}"
    )
    print(
        "artifacts "
        f"phase_path={result.phase_path} "
        f"model_path={result.model_path}"
    )
    print(
        "prediction "
        f"candidate_periods={list(result.candidate_periods)} "
        f"top_periods={list(result.top_periods)} "
        f"top_scores={[round(score, 4) for score in result.top_scores]}"
    )
    print(f"top1_period={result.top1_period}")

    if result.success and result.factors is not None and result.predicted_period is not None:
        print(f"selected_period={result.predicted_period}")
        print(f"{config.N} = {result.factors[0]} x {result.factors[1]}")
        return 0

    print("PH1 + DeepSet failed to recover non-trivial factors")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
