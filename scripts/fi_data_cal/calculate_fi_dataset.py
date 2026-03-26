from __future__ import annotations

import pickle
import sys
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
)

OUTPUT_PATH = Path("data/shared/fi_results.pkl")
NQUBIT_RANGE = range(4, 11)
SAMPLES = 64


def build_dataset(config: FiExperimentConfig) -> list[FiResult]:
    return calculate_fi_results(config)


def render_progress_bar(current: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return f"[{'#' * width}]"

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} ({ratio:.0%})"


def print_progress(current: int, total: int, config: FiExperimentConfig) -> None:
    layer_text = f", nlayer={config.nlayer}" if config.nlayer is not None else ""
    message = (
        f"\r{render_progress_bar(current, total)} "
        f"{config.circuit_type}, nqubit={config.nqubit}{layer_text}, repeat={config.repeat}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def load_dataset(input_path: Path) -> list[FiResult]:
    if not input_path.exists():
        return []

    try:
        with input_path.open("rb") as file_obj:
            data = pickle.load(file_obj)
    except (AttributeError, EOFError, OSError, pickle.PickleError, TypeError, ValueError):
        return []

    return data if isinstance(data, list) else []


def save_dataset(results: list[FiResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(results, file_obj)


def build_configs() -> list[FiExperimentConfig]:
    configs: list[FiExperimentConfig] = []

    for nqubit in NQUBIT_RANGE:
        configs.extend(
            [
                FiExperimentConfig(circuit_type="qft", nqubit=nqubit),
                FiExperimentConfig(circuit_type="ph1", nqubit=nqubit),
                FiExperimentConfig(
                    circuit_type="ph_1_random",
                    nqubit=nqubit,
                    repeat=SAMPLES,
                ),
            ]
        )
        configs.extend(
            FiExperimentConfig(
                circuit_type=circuit_type,
                nqubit=nqubit,
                nlayer=nlayer,
                repeat=SAMPLES,
            )
            for nlayer in range(1, nqubit)
            for circuit_type in ("ph_random", "ph_random_phase")
        )

    return configs


def main() -> None:
    existing_results = load_dataset(OUTPUT_PATH)
    configs = build_configs()
    total_configs = len(configs)
    new_results: list[FiResult] = []

    for index, config in enumerate(configs, start=1):
        print_progress(index - 1, total_configs, config)
        new_results.extend(build_dataset(config))
        print_progress(index, total_configs, config)

    existing_results.extend(new_results)
    save_dataset(existing_results, OUTPUT_PATH)
    print(f"saved {len(existing_results)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
