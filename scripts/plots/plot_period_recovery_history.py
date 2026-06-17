from __future__ import annotations

import argparse
from pathlib import Path

from altqft.plotting.period_recovery import (
    build_metric_series,
    load_history_payload,
    plot_history,
    plot_period_recovery_history,
    resolve_history_paths,
    resolve_nqubit_label,
    resolve_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one or more period-recovery training histories with epoch on the x-axis, "
            "loss on the left y-axis, and top-k accuracy on the right y-axis."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more history JSON paths. Defaults to data/period_recovery_{9,10,11}q_history.json "
            "when all three exist, otherwise the newest data/period_recovery_*_history.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output figure path or stem. Defaults to "
            "figs/recover/<history_stem>_<split>_metrics.<format>."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("svg", "pdf"),
        help="Output formats to write. Defaults to svg pdf.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which split to plot. Defaults to validation metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [str(output_format).lstrip(".") for output_format in args.formats]
    first_output = args.output
    if first_output is not None and formats:
        first_output = first_output.with_suffix(f".{formats[0]}")

    history_paths, output_path = plot_period_recovery_history(
        history_paths=args.history,
        output_path=first_output,
        split=str(args.split),
    )
    output_paths = [output_path]

    for output_format in formats[1:]:
        next_output_path = output_path.with_suffix(f".{output_format}")
        _, resolved_output_path = plot_period_recovery_history(
            history_paths=history_paths,
            output_path=next_output_path,
            split=str(args.split),
        )
        output_paths.append(resolved_output_path)

    print(
        "history_paths=" + ",".join(str(history_path) for history_path in history_paths)
    )
    print("output_paths=" + ",".join(str(path) for path in output_paths))


__all__ = [
    "build_metric_series",
    "load_history_payload",
    "plot_history",
    "plot_period_recovery_history",
    "resolve_history_paths",
    "resolve_nqubit_label",
    "resolve_output_path",
]


if __name__ == "__main__":
    main()
