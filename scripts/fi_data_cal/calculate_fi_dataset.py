from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
)
from altqft.nn.process_qc import available_cuda_device_count, resolve_compute_device

OUTPUT_PATH = Path("data/shared/fi_results.pkl")
NQUBIT_RANGE = range(10, 15)
SAMPLES = 64
PARALLEL_NQUBIT_THRESHOLD = 10
MAX_DEFAULT_CPU_WORKERS = 2


def build_dataset(config: FiExperimentConfig, device: str | None = None) -> list[FiResult]:
    return calculate_fi_results(config, device=device)


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
    except (
        AttributeError,
        EOFError,
        OSError,
        pickle.PickleError,
        TypeError,
        ValueError,
    ):
        return []

    return data if isinstance(data, list) else []


def save_dataset(results: list[FiResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(results, file_obj)


def resolve_nqubit_range(start: int, end: int) -> range:
    if start < 2:
        raise ValueError("nqubit start must be at least 2")
    if end < start:
        raise ValueError("nqubit end must be greater than or equal to start")
    return range(start, end + 1)


def build_configs(nqubit_range: range) -> list[FiExperimentConfig]:
    configs: list[FiExperimentConfig] = []

    for nqubit in nqubit_range:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Fisher-information dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination pickle path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=os.environ.get("ALTQFT_FI_DEVICE", "auto"),
        help="Execution device for FI calculation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_env_worker_count(),
        help="Override the worker count used for large-nqubit configs.",
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
    parser.add_argument(
        "--replace-range",
        action="store_true",
        help="Remove existing records for the selected qubit range before saving.",
    )
    return parser.parse_args()


def _env_worker_count() -> int | None:
    raw_value = os.environ.get("ALTQFT_FI_WORKERS")
    if raw_value is None:
        return None
    return int(raw_value)


def should_parallelize(configs: list[FiExperimentConfig]) -> bool:
    return any(config.nqubit >= PARALLEL_NQUBIT_THRESHOLD for config in configs)


def default_worker_count(total_configs: int, device_name: str) -> int:
    if total_configs <= 1:
        return 1

    if device_name == "cuda":
        gpu_count = available_cuda_device_count()
        if gpu_count > 1:
            return min(total_configs, gpu_count)
        return 1

    if device_name == "mps":
        return 1

    cpu_budget = min(os.cpu_count() or 1, MAX_DEFAULT_CPU_WORKERS)
    return min(total_configs, cpu_budget)


def resolve_worker_count(
    requested_workers: int | None,
    configs: list[FiExperimentConfig],
    device_name: str,
) -> int:
    if not should_parallelize(configs):
        return 1

    if requested_workers is None:
        return default_worker_count(len(configs), device_name)

    if requested_workers < 1:
        raise ValueError("workers must be at least 1")

    return requested_workers


def worker_device(worker_index: int, device_name: str) -> str:
    if device_name != "cuda":
        return device_name

    gpu_count = max(available_cuda_device_count(), 1)
    if gpu_count == 1:
        return "cuda"
    return f"cuda:{worker_index % gpu_count}"


def filter_results_by_nqubits(
    results: list[FiResult],
    nqubits: set[int],
) -> list[FiResult]:
    return [result for result in results if result.nqubit not in nqubits]


def run_serial(configs: list[FiExperimentConfig], device_name: str) -> list[FiResult]:
    total_configs = len(configs)
    new_results: list[FiResult] = []

    for index, config in enumerate(configs, start=1):
        print_progress(index - 1, total_configs, config)
        new_results.extend(build_dataset(config, device_name))
        print_progress(index, total_configs, config)

    return new_results


def run_parallel(
    configs: list[FiExperimentConfig],
    device_name: str,
    workers: int,
) -> list[FiResult]:
    total_configs = len(configs)
    ordered_chunks: list[list[FiResult] | None] = [None] * total_configs

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        future_map = {
            executor.submit(
                build_dataset,
                config,
                worker_device(index, device_name),
            ): (index, config)
            for index, config in enumerate(configs)
        }

        completed = 0
        for future in as_completed(future_map):
            index, config = future_map[future]
            ordered_chunks[index] = future.result()
            completed += 1
            print_progress(completed, total_configs, config)

    new_results: list[FiResult] = []
    for chunk in ordered_chunks:
        if chunk is None:
            raise RuntimeError("parallel FI task did not produce a result")
        new_results.extend(chunk)
    return new_results


def main() -> None:
    args = parse_args()
    resolved_device = resolve_compute_device(args.device)
    nqubit_range = resolve_nqubit_range(args.nqubit_start, args.nqubit_end)
    existing_results = load_dataset(args.output)
    configs = build_configs(nqubit_range)
    workers = resolve_worker_count(args.workers, configs, resolved_device)

    if args.replace_range:
        existing_results = filter_results_by_nqubits(
            existing_results,
            set(nqubit_range),
        )

    print(
        f"running {len(configs)} FI configs on {resolved_device} "
        f"with {workers} worker(s) for nqubits={nqubit_range.start}..{nqubit_range.stop - 1}"
    )

    if workers > 1:
        new_results = run_parallel(configs, resolved_device, workers)
    else:
        new_results = run_serial(configs, resolved_device)

    existing_results.extend(new_results)
    save_dataset(existing_results, args.output)
    print(f"saved {len(existing_results)} records to {args.output}")


if __name__ == "__main__":
    main()
