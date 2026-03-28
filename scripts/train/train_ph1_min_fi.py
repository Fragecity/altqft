from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from altqft.nn.train import (
    EpochResult,
    TrainConfig,
    serialize_config,
    train_model,
)
from altqft.nn.periods import build_default_period_range

NQUBIT_RANGE = range(8, 9)
EPOCHS = 300
LEARNING_RATE = 0.01
MONTE_CARLO_SAMPLES = 32
SEED = 42
LOG_INTERVAL = 25
SUMMARY_PATH = Path("data/shared/ph1_min_fi_summary.json")


def build_config(nqubit: int) -> TrainConfig:
    return TrainConfig(
        nqubit=nqubit,
        period_range=build_default_period_range(nqubit),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        monte_carlo_samples=MONTE_CARLO_SAMPLES,
        seed=SEED,
        log_interval=LOG_INTERVAL,
    )


def resolve_nqubit_range(start: int, end: int) -> range:
    if start < 2:
        raise ValueError("nqubit start must be at least 2")
    if end < start:
        raise ValueError("nqubit end must be greater than or equal to start")
    return range(start, end + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume optimized PH1 min-FI models.",
    )
    parser.add_argument(
        "--nqubit-start",
        type=int,
        default=NQUBIT_RANGE.start,
        help="Inclusive starting qubit count.",
    )
    parser.add_argument(
        "--nqubit-end",
        type=int,
        default=NQUBIT_RANGE.stop - 1,
        help="Inclusive ending qubit count.",
    )
    return parser.parse_args()


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


def history_matches_config(config: TrainConfig) -> bool:
    if not config.history_path.exists():
        return False

    payload = json.loads(config.history_path.read_text(encoding="utf-8"))
    stored_config = payload.get("config")
    return stored_config == serialize_config(config)


def build_summary_entry(
    config: TrainConfig, history: list[EpochResult]
) -> dict[str, object]:
    best_epoch = select_best_epoch(history)
    return {
        "nqubit": config.nqubit,
        "period_range": list(config.period_range),
        "best_epoch": asdict(best_epoch),
        "final_epoch": asdict(history[-1]),
        "model_path": str(config.model_path),
        "phase_path": str(config.phase_path),
        "history_path": str(config.history_path),
        "log_path": str(config.log_path),
    }


def load_existing_summary_results() -> list[dict[str, object]]:
    if not SUMMARY_PATH.exists():
        return []

    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"invalid summary payload: {SUMMARY_PATH}")

    summary_results: list[dict[str, object]] = []
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("nqubit"), int):
            summary_results.append(item)
    return summary_results


def save_summary(results: list[dict[str, object]], nqubit_range: range) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_results_by_nqubit = {
        entry["nqubit"]: entry for entry in load_existing_summary_results()
    }
    for entry in results:
        merged_results_by_nqubit[entry["nqubit"]] = entry

    payload = {
        "config": {
            "nqubit_range": list(nqubit_range),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "monte_carlo_samples": MONTE_CARLO_SAMPLES,
            "seed": SEED,
            "log_interval": LOG_INTERVAL,
        },
        "results": [
            merged_results_by_nqubit[nqubit]
            for nqubit in sorted(merged_results_by_nqubit)
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def has_completed_artifacts(config: TrainConfig) -> bool:
    required_paths = (
        config.model_path,
        config.phase_path,
        config.history_path,
        config.log_path,
    )
    return all(path.exists() for path in required_paths) and history_matches_config(
        config
    )


def train_or_resume(config: TrainConfig) -> tuple[list[EpochResult], bool]:
    if has_completed_artifacts(config):
        return load_history(config), True

    artifacts = train_model(config)
    return artifacts.history, False


def main() -> None:
    args = parse_args()
    nqubit_range = resolve_nqubit_range(args.nqubit_start, args.nqubit_end)
    summary_results: list[dict[str, object]] = []

    for nqubit in nqubit_range:
        config = build_config(nqubit)
        history, resumed = train_or_resume(config)
        summary_entry = build_summary_entry(config, history)
        summary_results.append(summary_entry)
        save_summary(summary_results, nqubit_range)

        best_epoch = summary_entry["best_epoch"]
        assert isinstance(best_epoch, dict)
        print(
            f"status={'resumed' if resumed else 'trained'} "
            f"nqubit={nqubit} "
            f"best_epoch={best_epoch['epoch']} "
            f"best_fi={float(best_epoch['min_fi']):.8f} "
            f"history_path={config.history_path}"
        )

    print(f"summary_path={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
