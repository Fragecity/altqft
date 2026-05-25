from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from altqft.circuits.HPgenerators import HP1_shared_parameter
from altqft.nn.train import (
    EpochResult,
    TrainConfig,
    serialize_config,
    train_model,
)
from altqft.nn.periods import build_default_period_range
from altqft.nn.process_qc import min_fi
from altqft.plotting.fi import HP1_SHARED_SUMMARY_FILE, plot_fi_dataset

NQUBIT_RANGE = range(7, 17)
EPOCHS = 800
LEARNING_RATE = 0.01
MONTE_CARLO_SAMPLES = 32
SEED = 42
LOG_INTERVAL = 25
MIN_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 200
EARLY_STOPPING_MIN_DELTA = 1e-5
SHIFT_INV_PERIOD_SAMPLES = 8
SHIFT_INV_SHIFT_SAMPLES = 8
SUMMARY_PATH = HP1_SHARED_SUMMARY_FILE
MIN_FI_SOURCE = "recomputed_from_phase_parameters"


class EpochSummary(TypedDict):
    epoch: int
    loss: float
    min_fi: float


class SummaryEntry(TypedDict):
    nqubit: int
    period_range: list[int]
    min_fi: float
    min_fi_source: str
    best_epoch: EpochSummary
    final_epoch: EpochSummary
    model_path: str
    phase_path: str
    history_path: str
    log_path: str


def build_config(nqubit: int) -> TrainConfig:
    return TrainConfig(
        nqubit=nqubit,
        period_range=build_default_period_range(nqubit),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        monte_carlo_samples=MONTE_CARLO_SAMPLES,
        seed=SEED,
        log_interval=LOG_INTERVAL,
        objective="hp1_shared_fi_shift",
        ansatz="HP1_shared",
        min_epochs=MIN_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        shift_inv_period_samples=SHIFT_INV_PERIOD_SAMPLES,
        shift_inv_shift_samples=SHIFT_INV_SHIFT_SAMPLES,
    )


def resolve_nqubit_range(start: int, end: int) -> range:
    if start < 2:
        raise ValueError("nqubit start must be at least 2")
    if end < start:
        raise ValueError("nqubit end must be greater than or equal to start")
    return range(start, end + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HP1_shared min-FI + shift-invariance models.",
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


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"invalid JSON payload: {path}")
    return payload


def _load_phase_values(path: Path) -> list[float]:
    payload = _load_json_object(path)
    phases = payload.get("phases")
    if not isinstance(phases, list) or not all(
        isinstance(value, (int, float)) for value in phases
    ):
        raise TypeError(f"invalid phase payload in {path}")
    return [float(value) for value in phases]


def recompute_trained_min_fi(config: TrainConfig) -> float:
    circuit = HP1_shared_parameter(
        config.nqubit,
        _load_phase_values(config.phase_path),
    )
    return min_fi(
        circuit,
        config.period_range,
        device=config.train_device,
    )


def _parse_epoch_result(item: object, path: Path) -> EpochResult:
    if not isinstance(item, dict):
        raise TypeError(f"invalid epoch payload in {path}")

    epoch = item.get("epoch")
    loss = item.get("loss")
    min_fi = item.get("min_fi")
    if not isinstance(epoch, int):
        raise TypeError(f"invalid epoch value in {path}")
    if not isinstance(loss, (int, float)) or not isinstance(min_fi, (int, float)):
        raise TypeError(f"invalid FI payload in {path}")

    return EpochResult(epoch=epoch, loss=float(loss), min_fi=float(min_fi))


def _epoch_summary(result: EpochResult) -> EpochSummary:
    if result.min_fi is None:
        raise ValueError("expected min_fi history for HP1_shared sweep")
    return {
        "epoch": result.epoch,
        "loss": result.loss,
        "min_fi": result.min_fi,
    }


def load_history(config: TrainConfig) -> list[EpochResult]:
    payload = _load_json_object(config.history_path)
    history_items = payload.get("history", [])
    if not isinstance(history_items, list):
        raise TypeError(f"invalid history payload: {config.history_path}")
    return [_parse_epoch_result(item, config.history_path) for item in history_items]


def history_matches_config(config: TrainConfig) -> bool:
    if not config.history_path.exists():
        return False

    payload = _load_json_object(config.history_path)
    stored_config = payload.get("config")
    return isinstance(stored_config, dict) and stored_config == serialize_config(config)


def build_summary_entry(
    config: TrainConfig, history: list[EpochResult]
) -> SummaryEntry:
    best_epoch = select_best_epoch(history)
    recomputed_min_fi = recompute_trained_min_fi(config)
    return {
        "nqubit": config.nqubit,
        "period_range": list(config.period_range),
        "min_fi": recomputed_min_fi,
        "min_fi_source": MIN_FI_SOURCE,
        "best_epoch": _epoch_summary(best_epoch),
        "final_epoch": _epoch_summary(history[-1]),
        "model_path": str(config.model_path),
        "phase_path": str(config.phase_path),
        "history_path": str(config.history_path),
        "log_path": str(config.log_path),
    }


def load_existing_summary_results() -> list[SummaryEntry]:
    if not SUMMARY_PATH.exists():
        return []

    payload = _load_json_object(SUMMARY_PATH)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"invalid summary payload: {SUMMARY_PATH}")

    summary_results: list[SummaryEntry] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        nqubit = item.get("nqubit")
        period_range = item.get("period_range")
        min_fi_value = item.get("min_fi")
        min_fi_source = item.get("min_fi_source")
        best_epoch = item.get("best_epoch")
        final_epoch = item.get("final_epoch")
        model_path = item.get("model_path")
        phase_path = item.get("phase_path")
        history_path = item.get("history_path")
        log_path = item.get("log_path")
        if not isinstance(nqubit, int):
            continue
        if not isinstance(period_range, list) or not all(
            isinstance(value, int) for value in period_range
        ):
            continue
        if not isinstance(best_epoch, dict) or not isinstance(final_epoch, dict):
            continue
        if not all(
            isinstance(path, str)
            for path in (model_path, phase_path, history_path, log_path)
        ):
            continue
        assert isinstance(model_path, str)
        assert isinstance(phase_path, str)
        assert isinstance(history_path, str)
        assert isinstance(log_path, str)
        best_epoch_summary = _epoch_summary(_parse_epoch_result(best_epoch, SUMMARY_PATH))
        final_epoch_summary = _epoch_summary(_parse_epoch_result(final_epoch, SUMMARY_PATH))
        parsed_min_fi = (
            float(min_fi_value)
            if isinstance(min_fi_value, (int, float))
            else best_epoch_summary["min_fi"]
        )

        summary_results.append(
            {
                "nqubit": nqubit,
                "period_range": period_range,
                "min_fi": parsed_min_fi,
                "min_fi_source": (
                    min_fi_source if isinstance(min_fi_source, str) else "legacy_best_epoch"
                ),
                "best_epoch": best_epoch_summary,
                "final_epoch": final_epoch_summary,
                "model_path": model_path,
                "phase_path": phase_path,
                "history_path": history_path,
                "log_path": log_path,
            }
        )
    return summary_results


def save_summary(results: list[SummaryEntry], nqubit_range: range) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_results_by_nqubit: dict[int, SummaryEntry] = {
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
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "shift_inv_period_samples": SHIFT_INV_PERIOD_SAMPLES,
            "shift_inv_shift_samples": SHIFT_INV_SHIFT_SAMPLES,
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


def save_summary_and_plot(
    results: list[SummaryEntry],
    nqubit_range: range,
) -> Path:
    save_summary(results, nqubit_range)
    return plot_fi_dataset(hp1_shared_summary_path=SUMMARY_PATH)


def main() -> None:
    args = parse_args()
    nqubit_range = resolve_nqubit_range(args.nqubit_start, args.nqubit_end)
    summary_results: list[SummaryEntry] = []

    for nqubit in nqubit_range:
        config = build_config(nqubit)
        history, resumed = train_or_resume(config)
        summary_entry = build_summary_entry(config, history)
        summary_results.append(summary_entry)
        plot_path = save_summary_and_plot(summary_results, nqubit_range)

        best_epoch = summary_entry["best_epoch"]
        print(
            f"status={'resumed' if resumed else 'trained'} "
            f"nqubit={nqubit} "
            f"best_epoch={best_epoch['epoch']} "
            f"min_fi={summary_entry['min_fi']:.8f} "
            f"min_fi_source={summary_entry['min_fi_source']} "
            f"plot_path={plot_path} "
            f"history_path={config.history_path}"
        )

    print(f"summary_path={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
