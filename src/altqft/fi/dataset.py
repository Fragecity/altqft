from __future__ import annotations

import os
import pickle
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from qiskit import QuantumCircuit

from altqft.circuits.HPgenerators import (
    HP1,
    HP1_random,
    HPrandom,
    HPrandom_phase,
    qft,
)
from altqft.nn.devices import available_cuda_device_count, resolve_compute_device
from altqft.nn.periods import build_default_period_range
from altqft.nn.process_qc import (
    min_fi,
)

CircuitFactory = Callable[[], QuantumCircuit]

OUTPUT_PATH = Path("data/shared/fi_results.pkl")
NQUBIT_RANGE = range(7, 19)
SAMPLES = 64
PARALLEL_NQUBIT_THRESHOLD = 10
MAX_DEFAULT_CPU_WORKERS = 2


@dataclass(frozen=True)
class FiExperimentConfig:
    circuit_type: str
    nqubit: int
    repeat: int = 1
    nlayer: int | None = None


@dataclass(frozen=True)
class FiResult:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


def _require_nlayer(circuit_type: str, nlayer: int | None) -> int:
    if nlayer is None:
        raise ValueError(f"{circuit_type} requires nlayer")
    return nlayer


def build_circuit(
    circuit_type: str,
    nqubit: int,
    nlayer: int | None = None,
) -> QuantumCircuit:
    circuit_key = circuit_type.lower()
    builders: dict[str, CircuitFactory] = {
        "qft": lambda: qft(nqubit),
        "ph1": lambda: HP1(nqubit),
        "hp1_random": lambda: HP1_random(nqubit),
        "hprandom": lambda: HPrandom(nqubit, _require_nlayer("HPrandom", nlayer)),
        "hprandom_phase": lambda: HPrandom_phase(
            nqubit,
            _require_nlayer("HPrandom_phase", nlayer),
        ),
    }

    if circuit_key not in builders:
        raise ValueError(f"unsupported circuit type: {circuit_type}")
    return builders[circuit_key]()


def default_period_range(nqubit: int) -> list[int]:
    return build_default_period_range(nqubit)


def calculate_fi_results(
    config: FiExperimentConfig,
    device: str | None = None,
) -> list[FiResult]:
    period_range = default_period_range(config.nqubit)
    return [
        FiResult(
            circuit_type=config.circuit_type,
            nqubit=config.nqubit,
            fi_value=min_fi(
                build_circuit(config.circuit_type, config.nqubit, config.nlayer),
                period_range=period_range,
                device=device,
            ),
            nlayer=config.nlayer,
        )
        for _ in range(config.repeat)
    ]


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
            for circuit_type in ("HPrandom", "HPrandom_phase")
        )

    return configs


def env_worker_count() -> int | None:
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


def remove_results_by_nqubits(
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


def run_serial_incremental(
    configs: list[FiExperimentConfig],
    device_name: str,
    existing_results: list[FiResult],
    output_path: Path,
) -> list[FiResult]:
    total_configs = len(configs)
    new_results: list[FiResult] = []

    for index, config in enumerate(configs, start=1):
        print_progress(index - 1, total_configs, config)
        chunk = build_dataset(config, device_name)
        new_results.extend(chunk)
        existing_results.extend(chunk)
        save_dataset(existing_results, output_path)
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


def build_fi_dataset(
    *,
    output_path: Path = OUTPUT_PATH,
    device: str = "auto",
    workers: int | None = None,
    nqubit_start: int = NQUBIT_RANGE.start,
    nqubit_end: int = NQUBIT_RANGE.stop - 1,
    replace_range: bool = False,
) -> list[FiResult]:
    resolved_device = resolve_compute_device(device)
    nqubit_range = resolve_nqubit_range(nqubit_start, nqubit_end)
    existing_results = load_dataset(output_path)
    configs = build_configs(nqubit_range)
    resolved_workers = resolve_worker_count(workers, configs, resolved_device)

    if replace_range:
        existing_results = remove_results_by_nqubits(
            existing_results,
            set(nqubit_range),
        )

    print(
        f"running {len(configs)} FI configs on {resolved_device} "
        f"with {resolved_workers} worker(s) for "
        f"nqubits={nqubit_range.start}..{nqubit_range.stop - 1}"
    )

    if resolved_workers > 1:
        new_results = run_parallel(configs, resolved_device, resolved_workers)
        existing_results.extend(new_results)
        save_dataset(existing_results, output_path)
    else:
        new_results = run_serial_incremental(
            configs,
            resolved_device,
            existing_results,
            output_path,
        )

    print(f"saved {len(existing_results)} records to {output_path}")
    return existing_results
