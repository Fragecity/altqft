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
            "One or more history JSON paths. Defaults to outputs/period_recovery_{9,10,11}q_history.json "
            "when all three exist, otherwise the newest outputs/period_recovery_*_history.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to figs/recover/<history_stem>_<split>.svg.",
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
    history_paths, output_path = plot_period_recovery_history(
        history_paths=args.history,
        output_path=args.output,
        split=str(args.split),
    )

    print(
        "history_paths=" + ",".join(str(history_path) for history_path in history_paths)
    )
    print(f"output_path={output_path}")


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
