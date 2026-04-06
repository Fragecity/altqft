from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from altqft.experiments.period_noise_9q10q import (
    DEFAULT_EXPERIMENT_ROOTS,
    DEFAULT_NOISE_LEVELS,
    DEFAULT_PERIOD_NOISE_RECIPE,
    ExperimentRoots,
    build_noise_levels,
    legacy_qubit_experiment_spec,
    run_period_noise_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain isolated 9q/10q period-recovery models with the successful 11q "
            "exact-shiftce recipe, then sweep held-out accuracy under global "
            "post-PH1 depolarizing noise."
        )
    )
    parser.add_argument(
        "--nqubit",
        type=int,
        nargs="+",
        default=[9, 10],
        help="One or more legacy-qubit runs to train and evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOTS.model_dir,
        help="Dedicated model root for this experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOTS.output_dir,
        help="Dedicated output root for this experiment.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOTS.dataset_dir,
        help="Dedicated shift-pool cache root for this experiment.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOTS.data_dir,
        help="Shared data root passed through to training utilities.",
    )
    parser.add_argument(
        "--noise-start",
        type=float,
        default=DEFAULT_NOISE_LEVELS[0],
        help="Largest global depolarizing strength in the sweep.",
    )
    parser.add_argument(
        "--noise-stop",
        type=float,
        default=DEFAULT_NOISE_LEVELS[-1],
        help="Smallest global depolarizing strength in the sweep.",
    )
    parser.add_argument(
        "--noise-count",
        type=int,
        default=len(DEFAULT_NOISE_LEVELS),
        help="Number of log-spaced noise points.",
    )
    parser.add_argument(
        "--variant-tag",
        type=str,
        default=DEFAULT_PERIOD_NOISE_RECIPE.variant_tag,
        help="Artifact tag appended to PH1 and period-net run names.",
    )
    parser.add_argument(
        "--ph1-train-device",
        type=str,
        default=DEFAULT_PERIOD_NOISE_RECIPE.ph1_train_device,
        help="PH1 optimization device: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--cache-device",
        type=str,
        default=DEFAULT_PERIOD_NOISE_RECIPE.cache_device,
        help="Shift-pool cache generation device: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_PERIOD_NOISE_RECIPE.seed,
        help="Shared RNG seed for training and evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = ExperimentRoots(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
    )
    recipe = replace(
        DEFAULT_PERIOD_NOISE_RECIPE,
        variant_tag=args.variant_tag,
        ph1_train_device=args.ph1_train_device,
        cache_device=args.cache_device,
        seed=args.seed,
    )
    noise_levels = build_noise_levels(
        start=args.noise_start,
        stop=args.noise_stop,
        count=args.noise_count,
    )
    summary = run_period_noise_experiment(
        qubit_specs=tuple(legacy_qubit_experiment_spec(value) for value in args.nqubit),
        roots=roots,
        recipe=recipe,
        noise_levels=noise_levels,
    )

    print(f"json_path={summary.json_path}")
    print(f"csv_path={summary.csv_path}")
    print(f"png_path={summary.png_path}")
    print(f"svg_path={summary.svg_path}")
    for result in summary.results:
        best_point = max(result.points, key=lambda point: point.accuracy)
        print(
            f"nqubit={result.nqubit} selected_epoch={result.selected_epoch} "
            f"selected_val_top1={result.selected_val_top1:.4f} "
            f"best_noise={best_point.noise_strength:.6f} best_accuracy={best_point.accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
