from __future__ import annotations

import argparse
from pathlib import Path

from altqft.nn.process_qc import resolve_compute_device
from calculate_fi_dataset import OUTPUT_PATH, load_dataset, save_dataset
from fisher_information_utils import FiExperimentConfig, FiResult, calculate_fi_results

DEFAULT_START_NQUBIT = 4
DEFAULT_END_NQUBIT = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute qft Fisher-information rows for nqubit 4..14 and replace them in the dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Existing FI dataset to update. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Destination pickle path. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--start-nqubit",
        type=int,
        default=DEFAULT_START_NQUBIT,
        help=f"First qft nqubit to recompute. Default: {DEFAULT_START_NQUBIT}",
    )
    parser.add_argument(
        "--end-nqubit",
        type=int,
        default=DEFAULT_END_NQUBIT,
        help=f"Last qft nqubit to recompute. Default: {DEFAULT_END_NQUBIT}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="Execution device for FI calculation. Default: cpu",
    )
    return parser.parse_args()


def should_replace_qft(result: FiResult, start_nqubit: int, end_nqubit: int) -> bool:
    return (
        result.circuit_type == "qft"
        and start_nqubit <= result.nqubit <= end_nqubit
    )


def build_qft_results(
    start_nqubit: int,
    end_nqubit: int,
    *,
    device: str,
) -> list[FiResult]:
    qft_results: list[FiResult] = []
    for nqubit in range(start_nqubit, end_nqubit + 1):
        config = FiExperimentConfig(circuit_type="qft", nqubit=nqubit)
        result_chunk = calculate_fi_results(config, device=device)
        if len(result_chunk) != 1:
            raise RuntimeError(f"expected exactly one qft row for nqubit={nqubit}, got {len(result_chunk)}")
        qft_results.extend(result_chunk)
    return qft_results


def sort_key(result: FiResult) -> tuple[int, str, int, float]:
    nlayer = -1 if result.nlayer is None else result.nlayer
    return (result.nqubit, result.circuit_type, nlayer, float(result.fi_value))


def main() -> None:
    args = parse_args()
    if args.start_nqubit > args.end_nqubit:
        raise ValueError("start-nqubit must be <= end-nqubit")

    resolved_device = resolve_compute_device(args.device)
    existing_results = load_dataset(args.input)

    preserved_results = [
        result
        for result in existing_results
        if not should_replace_qft(result, args.start_nqubit, args.end_nqubit)
    ]
    removed_count = len(existing_results) - len(preserved_results)

    recomputed_qft_results = build_qft_results(
        args.start_nqubit,
        args.end_nqubit,
        device=resolved_device,
    )
    merged_results = sorted(preserved_results + recomputed_qft_results, key=sort_key)
    save_dataset(merged_results, args.output)

    print(
        f"recomputed qft rows for nqubit={args.start_nqubit}..{args.end_nqubit} "
        f"using device={resolved_device}"
    )
    print(f"removed {removed_count} existing qft rows and wrote {len(recomputed_qft_results)} fresh qft rows")
    print(f"saved {len(merged_results)} total rows to {args.output}")
    for result in recomputed_qft_results:
        print(f"qft nqubit={result.nqubit} fi_value={float(result.fi_value):.10f}")


if __name__ == "__main__":
    main()
