from __future__ import annotations

import argparse
from pathlib import Path

from altqft.plotting.fi import (
    HP1_SHARED_SUMMARY_FILE,
    INPUT_FILE,
    OPTIMIZED_PH1_SUMMARY_FILE,
    OUTPUT_DIR,
    filter_results_by_nqubits,
    group_results_by_qubit,
    load_hp1_shared_results,
    load_optimized_ph1_results,
    load_results,
    plot_fi_dataset,
    plot_fi_vs_nqubits,
    resolve_nqubit_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FI datasets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input FI pickle file.",
    )
    parser.add_argument(
        "--optimized-summary",
        type=Path,
        default=OPTIMIZED_PH1_SUMMARY_FILE,
        help="Optimized PH1 summary JSON file.",
    )
    parser.add_argument(
        "--hp1-shared-summary",
        type=Path,
        default=HP1_SHARED_SUMMARY_FILE,
        help="HP1_shared summary JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--nqubit-start",
        type=int,
        default=None,
        help="Inclusive starting qubit count for plotting.",
    )
    parser.add_argument(
        "--nqubit-end",
        type=int,
        default=None,
        help="Inclusive ending qubit count for plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_fi_dataset(
        input_path=Path(args.input),
        optimized_summary_path=Path(args.optimized_summary),
        hp1_shared_summary_path=Path(args.hp1_shared_summary),
        output_dir=Path(args.output_dir),
        nqubit_start=args.nqubit_start,
        nqubit_end=args.nqubit_end,
    )
    print(f"output_path={output_path}")


__all__ = [
    "filter_results_by_nqubits",
    "group_results_by_qubit",
    "load_hp1_shared_results",
    "load_optimized_ph1_results",
    "load_results",
    "plot_fi_dataset",
    "plot_fi_vs_nqubits",
    "resolve_nqubit_filter",
]


if __name__ == "__main__":
    main()
