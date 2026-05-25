from __future__ import annotations

import argparse
import os
from pathlib import Path

from altqft.fi.dataset import (
    NQUBIT_RANGE,
    OUTPUT_PATH,
    FiExperimentConfig,
    FiResult,
    build_configs,
    build_dataset,
    build_fi_dataset,
    calculate_fi_results,
    default_worker_count,
    env_worker_count,
    load_dataset,
    remove_results_by_nqubits,
    resolve_nqubit_range,
    resolve_worker_count,
    run_parallel,
    run_serial,
    save_dataset,
    should_parallelize,
    worker_device,
)


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
        default=env_worker_count(),
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


def main() -> None:
    args = parse_args()
    build_fi_dataset(
        output_path=Path(args.output),
        device=str(args.device),
        workers=args.workers,
        nqubit_start=int(args.nqubit_start),
        nqubit_end=int(args.nqubit_end),
        replace_range=bool(args.replace_range),
    )


__all__ = [
    "FiExperimentConfig",
    "FiResult",
    "build_configs",
    "build_dataset",
    "calculate_fi_results",
    "default_worker_count",
    "load_dataset",
    "remove_results_by_nqubits",
    "resolve_nqubit_range",
    "resolve_worker_count",
    "run_parallel",
    "run_serial",
    "save_dataset",
    "should_parallelize",
    "worker_device",
]


if __name__ == "__main__":
    main()
