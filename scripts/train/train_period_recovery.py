from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from altqft.nn.optimized_ph1 import ensure_optimized_ph1
from altqft.nn.period_recovery import (
    PeriodRecoveryDatasetConfig,
    PeriodRecoveryTrainConfig,
    generate_period_recovery_dataset,
    train_period_recovery,
)

DEFAULT_NQUBIT = 10
DEFAULT_TOP_K = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 300
DEFAULT_NUM_TRAIN_SAMPLES = 1440
DEFAULT_NUM_VAL_SAMPLES = 360
DEFAULT_SEED = 7
DEFAULT_MODEL_DIR = Path("model")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_DATASET_SUBDIR = "period_recovery"
DEFAULT_FI_EPOCHS = 1000
DEFAULT_LOG_INTERVAL = 10
DEFAULT_FI_LOG_INTERVAL = 10
DEFAULT_MIN_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 75


def default_measurement_count(nqubit: int) -> int:
    return 1024 * nqubit**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a classical period-recovery model from bitmatrix measurements. "
            "By default this reuses an existing optimized PH1 phase artifact if present."
        ),
    )
    parser.add_argument(
        "--nqubit", type=int, default=DEFAULT_NQUBIT, help="Qubit count."
    )
    parser.add_argument(
        "--measurement-count",
        type=int,
        default=None,
        help="Measurements per bitmatrix sample. Defaults to 1024 * nqubit^2.",
    )
    parser.add_argument(
        "--period-min",
        type=int,
        default=2,
        help="Smallest candidate period to include in training labels.",
    )
    parser.add_argument(
        "--period-max",
        type=int,
        default=None,
        help="Largest candidate period to include in training labels.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=DEFAULT_NUM_TRAIN_SAMPLES,
        help="Number of training bitmatrices to generate.",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=DEFAULT_NUM_VAL_SAMPLES,
        help="Number of validation bitmatrices to generate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=range(1, 5),
        default=DEFAULT_TOP_K,
        help="Top-k metric to track; the 4-bit decoder retains four beams.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum classifier epochs."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Classifier learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Classifier Adam weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout used in the classifier head.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for multiclass classification.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help="Minimum classifier epochs before early stopping is allowed.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Number of stale epochs to tolerate before early stopping.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Epoch logging interval for classifier training.",
    )
    parser.add_argument(
        "--fi-epochs",
        type=int,
        default=DEFAULT_FI_EPOCHS,
        help="Epochs to use only when PH1 must be optimized or re-optimized.",
    )
    parser.add_argument(
        "--fi-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate to use only when PH1 must be optimized or re-optimized.",
    )
    parser.add_argument(
        "--fi-log-interval",
        type=int,
        default=DEFAULT_FI_LOG_INTERVAL,
        help="Epoch logging interval for PH1 optimization.",
    )
    parser.add_argument(
        "--fi-train-device",
        type=str,
        default="auto",
        help="PH1 training device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for model artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Base data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for logs and training histories.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Explicit dataset directory. Defaults to <data-dir>/period_recovery.",
    )
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="Rebuild bitmatrix train/val datasets even if cached copies exist.",
    )
    parser.add_argument(
        "--force-reoptimize-phases",
        action="store_true",
        help="Ignore existing PH1 phase artifacts and retrain the quantum phase model.",
    )
    parser.add_argument(
        "--disable-stratified-period-sampling",
        action="store_true",
        help="Use unconstrained random period sampling instead of stratified sampling.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("flat", "shift_pool"),
        default="flat",
        help="Dataset cache mode.",
    )
    parser.add_argument(
        "--exact-support",
        action="store_true",
        help="Use exact Shor support when generating PH1 probability distributions.",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=1,
        help="Pool size multiplier for shift-pool datasets.",
    )
    parser.add_argument(
        "--held-out-shifts-per-period",
        type=int,
        default=1,
        help="How many shifts per period to reserve for validation in shift-pool mode.",
    )
    parser.add_argument(
        "--val-draws-per-heldout-shift",
        type=int,
        default=4,
        help="Deterministic validation draws per held-out shift in shift-pool mode.",
    )
    parser.add_argument(
        "--train-draws-per-epoch",
        type=int,
        default=None,
        help="Number of random train draws per epoch in shift-pool mode. Defaults to one draw per train shift pool.",
    )
    parser.add_argument(
        "--variant-tag",
        type=str,
        default=None,
        help="Optional artifact variant tag appended to PH1, dataset, and DeepSet artifacts.",
    )
    parser.add_argument(
        "--cache-device",
        type=str,
        default="cpu",
        help="Device used when computing cached shift-pool distributions: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--fi-objective",
        choices=("min_fi", "shift_ce_mean"),
        default="min_fi",
        help="Objective used when optimizing PH1 phases.",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Build or refresh the cached dataset, then exit without training the DeepSet model.",
    )
    return parser.parse_args()


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return cast(Path, args.dataset_dir)
    return cast(Path, args.data_dir) / DEFAULT_DATASET_SUBDIR


def build_configs(
    args: argparse.Namespace,
) -> tuple[PeriodRecoveryTrainConfig, PeriodRecoveryDatasetConfig]:
    measurement_count = (
        args.measurement_count
        if args.measurement_count is not None
        else default_measurement_count(args.nqubit)
    )
    num_train_samples = 0 if args.dataset_mode == "shift_pool" else args.num_train_samples
    num_val_samples = 0 if args.dataset_mode == "shift_pool" else args.num_val_samples
    dataset_dir = resolve_dataset_dir(args)
    train_config = PeriodRecoveryTrainConfig(
        nqubit=args.nqubit,
        period_min=args.period_min,
        period_max=args.period_max,
        top_k=args.top_k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        min_epochs=args.min_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        log_interval=args.log_interval,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force_reoptimize_phases=args.force_reoptimize_phases,
        regenerate_dataset=args.regenerate_dataset,
        fi_epochs=args.fi_epochs,
        fi_learning_rate=args.fi_learning_rate,
        fi_log_interval=args.fi_log_interval,
        fi_objective=args.fi_objective,
        fi_exact_support=args.exact_support,
        fi_train_device=args.fi_train_device,
        dataset_mode=args.dataset_mode,
        variant_tag=args.variant_tag,
    )
    dataset_config = PeriodRecoveryDatasetConfig(
        nqubit=args.nqubit,
        measurement_count=measurement_count,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        period_min=args.period_min,
        period_max=args.period_max,
        seed=args.seed,
        stratify_periods=not args.disable_stratified_period_sampling,
        dataset_dir=dataset_dir,
        exact_support=args.exact_support,
        cache_mode=args.dataset_mode,
        pool_multiplier=args.pool_multiplier,
        held_out_shifts_per_period=args.held_out_shifts_per_period,
        val_draws_per_heldout_shift=args.val_draws_per_heldout_shift,
        train_draws_per_epoch=args.train_draws_per_epoch,
        variant_tag=args.variant_tag,
        cache_device=args.cache_device,
    )
    return train_config, dataset_config


def main() -> None:
    args = parse_args()
    train_config, dataset_config = build_configs(args)

    optimized_ph1 = ensure_optimized_ph1(
        args.nqubit,
        period_range=dataset_config.candidate_periods,
        epochs=train_config.fi_epochs,
        learning_rate=train_config.fi_learning_rate,
        seed=train_config.seed,
        log_interval=train_config.fi_log_interval,
        model_dir=train_config.model_dir,
        data_dir=train_config.data_dir,
        output_dir=train_config.output_dir,
        force_reoptimize=train_config.force_reoptimize_phases,
        objective=train_config.fi_objective,
        exact_support=train_config.fi_exact_support,
        variant_tag=train_config.variant_tag,
        train_device=train_config.fi_train_device,
    )
    dataset_artifacts = generate_period_recovery_dataset(
        dataset_config,
        optimized_ph1,
        regenerate=train_config.regenerate_dataset,
    )
    if args.dataset_only:
        print(
            f"HPstatus={'reused' if optimized_ph1.reused_existing else 'trained'} "
            f"phase_path={optimized_ph1.phase_path}"
        )
        print(f"period_range={dataset_config.candidate_periods}")
        print(
            f"dataset_mode={dataset_artifacts.cache_mode} "
            f"train={dataset_artifacts.train_path} "
            f"val={dataset_artifacts.val_path}"
        )
        if dataset_artifacts.manifest_path is not None:
            print(f"manifest_path={dataset_artifacts.manifest_path}")
        return
    artifacts = train_period_recovery(train_config, dataset_artifacts, optimized_ph1)

    print(
        f"HPstatus={'reused' if optimized_ph1.reused_existing else 'trained'} "
        f"phase_path={optimized_ph1.phase_path}"
    )
    print(f"period_range={dataset_config.candidate_periods}")
    print(
        f"dataset train={dataset_artifacts.train_path} val={dataset_artifacts.val_path}"
    )
    print(
        f"selected_epoch={artifacts.selected_epoch} "
        f"selected_top1={artifacts.selected_val_top1:.4f} "
        f"selected_top{artifacts.top_k}={artifacts.selected_val_topk:.4f} "
        f"stopped_early={artifacts.stopped_early}"
    )
    print(
        f"model_path={artifacts.model_path} history_path={artifacts.history_path} "
        f"log_path={artifacts.log_path}"
    )


if __name__ == "__main__":
    main()
