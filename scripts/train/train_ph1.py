from __future__ import annotations

import argparse
from pathlib import Path

from altqft.nn.optimized_ph1 import ensure_optimized_ph1
from altqft.nn.periods import build_period_range

DEFAULT_NQUBIT = 11
DEFAULT_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_SEED = 7
DEFAULT_LOG_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or reuse an optimized PH1 artifact.")
    parser.add_argument("--nqubit", type=int, default=DEFAULT_NQUBIT, help="Qubit count.")
    parser.add_argument(
        "--period-min",
        type=int,
        default=None,
        help="Smallest candidate period. Defaults to the 11q legacy lower bound when omitted.",
    )
    parser.add_argument(
        "--period-max",
        type=int,
        default=None,
        help="Largest candidate period.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Epoch logging interval.",
    )
    parser.add_argument(
        "--objective",
        choices=("min_fi", "shift_ce_mean"),
        default="min_fi",
        help="PH1 optimization objective.",
    )
    parser.add_argument(
        "--exact-support",
        action="store_true",
        help="Use exact support when computing PH1 objective distributions.",
    )
    parser.add_argument(
        "--variant-tag",
        type=str,
        default=None,
        help="Optional artifact variant tag appended to the PH1 run name.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Model directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--train-device",
        type=str,
        default="auto",
        help="PH1 training device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--force-reoptimize",
        action="store_true",
        help="Ignore any matching cached artifact and retrain PH1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_period_min = int(args.period_min) if args.period_min is not None else int(args.nqubit)
    period_range = build_period_range(
        int(args.nqubit),
        min_period=resolved_period_min,
        max_period=args.period_max,
    )
    artifact = ensure_optimized_ph1(
        int(args.nqubit),
        period_range=period_range,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        log_interval=int(args.log_interval),
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        force_reoptimize=bool(args.force_reoptimize),
        objective=str(args.objective),
        exact_support=bool(args.exact_support),
        variant_tag=args.variant_tag,
        train_device=str(args.train_device),
    )
    print(
        f"status={'reused' if artifact.reused_existing else 'trained'} "
        f"objective={artifact.objective} "
        f"exact_support={artifact.exact_support} "
        f"variant_tag={artifact.variant_tag}"
    )
    print(f"period_range={artifact.period_range}")
    print(f"phase_path={artifact.phase_path}")
    print(f"log_path={artifact.log_path}")
    print(
        f"final_loss={artifact.final_loss} "
        f"final_min_fi={artifact.final_min_fi} "
        f"final_mean_shift_l1={artifact.final_mean_shift_l1}"
    )


if __name__ == "__main__":
    main()
