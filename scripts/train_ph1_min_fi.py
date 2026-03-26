from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from altqft.nn.train import EpochResult, TrainArtifacts, TrainConfig, build_default_period_range, train_model

NQUBIT_RANGE = range(4, 13)
EPOCHS = 300
LEARNING_RATE = 0.05
SEED = 7
LOG_INTERVAL = 25
SUMMARY_PATH = Path("data/shared/ph1_min_fi_summary.json")


def build_config(nqubit: int) -> TrainConfig:
    return TrainConfig(
        nqubit=nqubit,
        period_range=build_default_period_range(nqubit),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        log_interval=LOG_INTERVAL,
    )


def select_best_epoch(history: list[EpochResult]) -> EpochResult:
    if not history:
        raise RuntimeError("training history is empty")
    return min(history, key=lambda item: item.loss)


def build_summary_entry(config: TrainConfig, artifacts: TrainArtifacts) -> dict[str, object]:
    best_epoch = select_best_epoch(artifacts.history)
    return {
        "nqubit": config.nqubit,
        "best_epoch": asdict(best_epoch),
        "final_epoch": asdict(artifacts.history[-1]),
        "model_path": str(artifacts.model_path),
        "phase_path": str(artifacts.phase_path),
        "history_path": str(artifacts.history_path),
        "log_path": str(artifacts.log_path),
    }


def save_summary(results: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "nqubit_range": list(NQUBIT_RANGE),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "log_interval": LOG_INTERVAL,
        },
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    summary_results: list[dict[str, object]] = []

    for nqubit in NQUBIT_RANGE:
        config = build_config(nqubit)
        artifacts = train_model(config)
        summary_entry = build_summary_entry(config, artifacts)
        summary_results.append(summary_entry)

        best_epoch = summary_entry["best_epoch"]
        assert isinstance(best_epoch, dict)
        print(
            f"nqubit={nqubit} "
            f"best_epoch={best_epoch['epoch']} "
            f"best_fi={-float(best_epoch['loss']):.8f} "
            f"history_path={artifacts.history_path}"
        )

    save_summary(summary_results)
    print(f"summary_path={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
