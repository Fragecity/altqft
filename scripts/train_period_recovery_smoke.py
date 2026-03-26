from __future__ import annotations

import logging
from pathlib import Path

from altqft.nn.optimized_ph1 import ensure_optimized_ph1
from altqft.nn.period_recovery import (
    PeriodRecoveryDatasetConfig,
    PeriodRecoveryTrainConfig,
    generate_period_recovery_dataset,
    train_period_recovery,
)

LOGGER_NAME = "scripts.train_period_recovery_smoke"
NQUBIT = 10
MEASUREMENT_COUNT = 1024 * NQUBIT**2
NUM_TRAIN_SAMPLES = 16
NUM_VAL_SAMPLES = 8
BATCH_SIZE = 4
TOP_K = 3
FI_EPOCHS = 1000
CLASSIFIER_EPOCHS = 1000
SEED = 7
MODEL_DIR = Path("model")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATASET_DIR = DATA_DIR / "period_recovery"
SCRIPT_LOG_PATH = OUTPUT_DIR / "train_period_recovery_smoke.log"


def configure_script_logger() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(SCRIPT_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    logger = configure_script_logger()
    train_config = PeriodRecoveryTrainConfig(
        nqubit=NQUBIT,
        top_k=TOP_K,
        batch_size=BATCH_SIZE,
        epochs=CLASSIFIER_EPOCHS,
        seed=SEED,
        force_reoptimize_phases=True,
        regenerate_dataset=True,
        fi_epochs=FI_EPOCHS,
        log_interval=1,
        fi_log_interval=1,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )
    dataset_config = PeriodRecoveryDatasetConfig(
        nqubit=NQUBIT,
        measurement_count=MEASUREMENT_COUNT,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_val_samples=NUM_VAL_SAMPLES,
        seed=SEED,
        dataset_dir=DATASET_DIR,
    )

    optimized_ph1 = ensure_optimized_ph1(
        NQUBIT,
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

    logger.info(
        "phase_path=%s train_dataset=%s val_dataset=%s",
        optimized_ph1.phase_path,
        dataset_artifacts.train_path,
        dataset_artifacts.val_path,
    )
    logger.info(
        "top1=%.4f top%d=%.4f model_path=%s history_path=%s",
        artifacts.final_val_top1,
        artifacts.top_k,
        artifacts.final_val_topk,
        artifacts.model_path,
        artifacts.history_path,
    )


if __name__ == "__main__":
    main()
