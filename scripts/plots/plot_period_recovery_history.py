from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_HISTORY_DIR = Path("outputs")
DEFAULT_FIGURE_DIR = Path("figs/recover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot period-recovery training history with epoch on the x-axis, "
            "loss on the left y-axis, and top-k accuracy on the right y-axis."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="History JSON path. Defaults to the newest outputs/period_recovery_*_history.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to figs/recover/<history_stem>_<split>.png.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which split to plot. Defaults to validation metrics.",
    )
    return parser.parse_args()


def resolve_history_path(candidate: Path | None) -> Path:
    if candidate is not None:
        return candidate

    matches = sorted(
        DEFAULT_HISTORY_DIR.glob("period_recovery_*_history.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No period recovery history JSON found under outputs/.")
    return matches[0]


def load_history_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {path}")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"No epoch history found in {path}")
    return payload


def resolve_output_path(history_path: Path, split: str, candidate: Path | None) -> Path:
    if candidate is not None:
        return candidate
    stem = history_path.stem.removesuffix("_history")
    return DEFAULT_FIGURE_DIR / f"{stem}_{split}_metrics.png"


def build_metric_series(payload: dict[str, Any], split: str) -> tuple[list[int], list[float], list[float], list[float], int]:
    history = payload["history"]
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("History config must be a JSON object")

    top_k = config.get("top_k", 3)
    if not isinstance(top_k, int):
        raise ValueError("config.top_k must be an integer")

    epochs: list[int] = []
    losses: list[float] = []
    top1_values: list[float] = []
    topk_values: list[float] = []

    loss_key = f"{split}_loss"
    top1_key = f"{split}_top1"
    topk_key = f"{split}_topk"

    for item in history:
        if not isinstance(item, dict):
            raise ValueError("Each history item must be a JSON object")
        epoch = item.get("epoch")
        loss = item.get(loss_key)
        top1 = item.get(top1_key)
        topk = item.get(topk_key)
        if not isinstance(epoch, int):
            raise ValueError("Epoch must be an integer")
        if not isinstance(loss, (int, float)):
            raise ValueError(f"Missing numeric field {loss_key} in epoch={epoch}")
        if not isinstance(top1, (int, float)):
            raise ValueError(f"Missing numeric field {top1_key} in epoch={epoch}")
        if not isinstance(topk, (int, float)):
            raise ValueError(f"Missing numeric field {topk_key} in epoch={epoch}")

        epochs.append(epoch)
        losses.append(float(loss))
        top1_values.append(float(top1))
        topk_values.append(float(topk))

    return epochs, losses, top1_values, topk_values, top_k


def plot_history(
    history_path: Path,
    output_path: Path,
    split: str,
    epochs: list[int],
    losses: list[float],
    top1_values: list[float],
    topk_values: list[float],
    top_k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(
        epochs,
        losses,
        color="#1f77b4",
        linewidth=2.2,
        label=f"{split} loss",
    )[0]
    top1_line = ax_acc.plot(
        epochs,
        top1_values,
        color="#d62728",
        linewidth=2.0,
        label=f"{split} top1",
    )[0]
    topk_line = ax_acc.plot(
        epochs,
        topk_values,
        color="#2ca02c",
        linewidth=2.0,
        label=f"{split} top{top_k}",
    )[0]

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color=loss_line.get_color())
    ax_acc.set_ylabel("Accuracy", color=top1_line.get_color())
    ax_loss.tick_params(axis="y", labelcolor=loss_line.get_color())
    ax_acc.tick_params(axis="y", labelcolor=top1_line.get_color())
    ax_acc.set_ylim(0.0, 1.0)
    ax_loss.grid(True, alpha=0.25)

    title_stem = history_path.stem.removesuffix("_history")
    ax_loss.set_title(f"{title_stem} ({split} metrics)")

    lines = [loss_line, top1_line, topk_line]
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="upper right")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = resolve_history_path(args.history)
    payload = load_history_payload(history_path)
    epochs, losses, top1_values, topk_values, top_k = build_metric_series(payload, args.split)
    output_path = resolve_output_path(history_path, args.split, args.output)

    plot_history(
        history_path=history_path,
        output_path=output_path,
        split=args.split,
        epochs=epochs,
        losses=losses,
        top1_values=top1_values,
        topk_values=topk_values,
        top_k=top_k,
    )

    print(f"history_path={history_path}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
