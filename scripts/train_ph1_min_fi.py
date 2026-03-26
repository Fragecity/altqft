from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from altqft.nn.train import EpochResult, TrainConfig, build_default_period_range, train_model

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


def load_history(config: TrainConfig) -> list[EpochResult]:
    payload = json.loads(config.history_path.read_text(encoding="utf-8"))
    history_items = payload.get("history", [])
    if not isinstance(history_items, list):
        raise TypeError(f"invalid history payload: {config.history_path}")
    return [EpochResult(**item) for item in history_items]


def build_summary_entry(config: TrainConfig, history: list[EpochResult]) -> dict[str, object]:
    best_epoch = select_best_epoch(history)
    return {
        "nqubit": config.nqubit,
        "best_epoch": asdict(best_epoch),
        "final_epoch": asdict(history[-1]),
        "model_path": str(config.model_path),
        "phase_path": str(config.phase_path),
        "history_path": str(config.history_path),
        "log_path": str(config.log_path),
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


def has_completed_artifacts(config: TrainConfig) -> bool:
    required_paths = (config.model_path, config.phase_path, config.history_path, config.log_path)
    return all(path.exists() for path in required_paths)


def train_or_resume(config: TrainConfig) -> tuple[list[EpochResult], bool]:
    if has_completed_artifacts(config):
        return load_history(config), True

    artifacts = train_model(config)
    return artifacts.history, False


def main() -> None:
    summary_results: list[dict[str, object]] = []

    for nqubit in NQUBIT_RANGE:
        config = build_config(nqubit)
        history, resumed = train_or_resume(config)
        summary_entry = build_summary_entry(config, history)
        summary_results.append(summary_entry)
        save_summary(summary_results)

        best_epoch = summary_entry["best_epoch"]
        assert isinstance(best_epoch, dict)
        print(
            f"status={'resumed' if resumed else 'trained'} "
            f"nqubit={nqubit} "
            f"best_epoch={best_epoch['epoch']} "
            f"best_fi={-float(best_epoch['loss']):.8f} "
            f"history_path={config.history_path}"
        )

    print(f"summary_path={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
