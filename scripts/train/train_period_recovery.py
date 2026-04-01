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
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 300
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
    parser.add_argument("--nqubit", type=int, default=DEFAULT_NQUBIT, help="Qubit count.")
    parser.add_argument(
        "--measurement-count",
        type=int,
        default=None,
        help="Measurements per bitmatrix sample. Defaults to 1024 * nqubit^2.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=256,
        help="Number of training bitmatrices to generate.",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=64,
        help="Number of validation bitmatrices to generate.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k metric to track.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum classifier epochs.")
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
    dataset_dir = resolve_dataset_dir(args)
    train_config = PeriodRecoveryTrainConfig(
        nqubit=args.nqubit,
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
    )
    dataset_config = PeriodRecoveryDatasetConfig(
        nqubit=args.nqubit,
        measurement_count=measurement_count,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        seed=args.seed,
        stratify_periods=not args.disable_stratified_period_sampling,
        dataset_dir=dataset_dir,
    )
    return train_config, dataset_config


def main() -> None:
    args = parse_args()
    train_config, dataset_config = build_configs(args)

    optimized_ph1 = ensure_optimized_ph1(
        args.nqubit,
        epochs=train_config.fi_epochs,
        learning_rate=train_config.fi_learning_rate,
        seed=train_config.seed,
        log_interval=train_config.fi_log_interval,
        model_dir=train_config.model_dir,
        data_dir=train_config.data_dir,
        output_dir=train_config.output_dir,
        force_reoptimize=train_config.force_reoptimize_phases,
    )
    dataset_artifacts = generate_period_recovery_dataset(
        dataset_config,
        optimized_ph1,
        regenerate=train_config.regenerate_dataset,
    )
    artifacts = train_period_recovery(train_config, dataset_artifacts, optimized_ph1)

    print(
        f"ph_status={'reused' if optimized_ph1.reused_existing else 'trained'} "
        f"phase_path={optimized_ph1.phase_path}"
    )
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
