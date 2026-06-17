from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_HISTORY_DIR = Path("data")
DEFAULT_FIGURE_DIR = Path("figs/recover")
DEFAULT_OUTPUT_SUFFIX = ".svg"
DEFAULT_COMBINED_NQUBITS = (9, 10, 11)
DEFAULT_MAX_EPOCH = 0
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 21
LEGEND_FONT_SIZE = 21
TICK_FONT_SIZE = 18
GROUP_COLORS: dict[int, dict[str, str]] = {
    9: {
        "loss": "#1d4ed8",
        "top1": "#3b82f6",
        "topk": "#93c5fd",
    },
    10: {
        "loss": "#be185d",
        "top1": "#ec4899",
        "topk": "#f9a8d4",
    },
    11: {
        "loss": "#6d28d9",
        "top1": "#8b5cf6",
        "topk": "#c4b5fd",
    },
    18: {
        "loss": "#1d4ed8",
        "top1": "#3b82f6",
        "topk": "#93c5fd",
    },
}
FALLBACK_METRIC_COLORS = {
    "loss": "#1f77b4",
    "top1": "#d62728",
    "topk": "#2ca02c",
}
METRIC_LINESTYLES = {
    "loss": "solid",
    "top1": "--",
    "topk": "-.",
}


def _default_combined_history_paths() -> list[Path] | None:
    paths = [
        DEFAULT_HISTORY_DIR / f"period_recovery_{nqubit}q_history.json"
        for nqubit in DEFAULT_COMBINED_NQUBITS
    ]
    if all(path.exists() for path in paths):
        return paths
    return None


def resolve_history_paths(candidates: Sequence[Path] | None) -> list[Path]:
    if candidates is not None:
        return [Path(candidate) for candidate in candidates]

    default_combined = _default_combined_history_paths()
    if default_combined is not None:
        return default_combined

    matches = sorted(
        DEFAULT_HISTORY_DIR.glob("period_recovery_*_history.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No period recovery history JSON found under outputs/.")
    return [matches[0]]


def load_history_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {path}")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"No epoch history found in {path}")
    return payload


def resolve_output_path(
    history_paths: Sequence[Path],
    split: str,
    candidate: Path | None,
) -> Path:
    if candidate is not None:
        if candidate.suffix:
            return candidate
        return candidate.with_suffix(DEFAULT_OUTPUT_SUFFIX)
    stems = [
        history_path.stem.removesuffix("_history") for history_path in history_paths
    ]
    if len(stems) == 1:
        stem = stems[0]
    else:
        stem = "_".join(stems)
    return DEFAULT_FIGURE_DIR / f"{stem}_{split}_metrics{DEFAULT_OUTPUT_SUFFIX}"


def build_metric_series(
    payload: dict[str, Any],
    split: str,
) -> tuple[list[int], list[float], list[float], list[float], int]:
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
    topk_key_candidates = (f"{split}_top{top_k}", f"{split}_topk")

    for item in history:
        if not isinstance(item, dict):
            raise ValueError("Each history item must be a JSON object")
        epoch = item.get("epoch")
        loss = item.get(loss_key)
        top1 = item.get(top1_key)
        topk_key = next(
            (candidate for candidate in topk_key_candidates if candidate in item),
            topk_key_candidates[0],
        )
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

    if DEFAULT_MAX_EPOCH > 0:
        selected_indices = [
            index for index, epoch in enumerate(epochs) if epoch <= DEFAULT_MAX_EPOCH
        ]
        epochs = [epochs[index] for index in selected_indices]
        losses = [losses[index] for index in selected_indices]
        top1_values = [top1_values[index] for index in selected_indices]
        topk_values = [topk_values[index] for index in selected_indices]

    return epochs, losses, top1_values, topk_values, top_k


def resolve_nqubit_label(
    payload: dict[str, Any],
    history_path: Path,
) -> tuple[int | None, str]:
    config = payload.get("config", {})
    if isinstance(config, dict):
        nqubit = config.get("nqubit")
        if isinstance(nqubit, int):
            return nqubit, f"{nqubit}q"

    stem = history_path.stem.removesuffix("_history")
    parts = stem.split("_")
    nqubit_token = parts[-1] if parts else ""
    if nqubit_token.endswith("q") and nqubit_token[:-1].isdigit():
        nqubit = int(nqubit_token[:-1])
        return nqubit, f"{nqubit}q"
    return None, stem


def plot_history(
    histories: Sequence[tuple[Path, dict[str, Any]]],
    output_path: Path,
    split: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(11.5, 6.05), constrained_layout=True)
    ax_acc = ax_loss.twinx()
    legend_lines: list[Line2D] = []

    for history_path, payload in histories:
        epochs, losses, top1_values, topk_values, top_k = build_metric_series(
            payload,
            split,
        )
        nqubit, run_label = resolve_nqubit_label(payload, history_path)
        palette = (
            GROUP_COLORS.get(nqubit, FALLBACK_METRIC_COLORS)
            if nqubit is not None
            else FALLBACK_METRIC_COLORS
        )

        loss_line = ax_loss.plot(
            epochs,
            losses,
            color=palette["loss"],
            linewidth=2.2,
            linestyle=METRIC_LINESTYLES["loss"],
            label=f"{run_label} loss",
        )[0]
        top1_line = ax_acc.plot(
            epochs,
            top1_values,
            color=palette["top1"],
            linewidth=2.0,
            linestyle=METRIC_LINESTYLES["top1"],
            label=f"{run_label} top1 accuracy",
        )[0]
        topk_line = ax_acc.plot(
            epochs,
            topk_values,
            color=palette["topk"],
            linewidth=2.0,
            linestyle=METRIC_LINESTYLES["topk"],
            label=f"{run_label} top{top_k} accuracy",
        )[0]
        legend_lines.extend((loss_line, top1_line, topk_line))

    ax_loss.set_xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    ax_loss.set_ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    ax_acc.set_ylabel("Accuracy", fontsize=LABEL_FONT_SIZE)
    ax_acc.set_ylim(0.0, 1.0)
    loss_ymin, loss_ymax = ax_loss.get_ylim()
    acc_ymin, acc_ymax = ax_acc.get_ylim()
    ax_loss.set_ylim(loss_ymin, loss_ymax * 1.1)
    ax_acc.set_ylim(acc_ymin, acc_ymax * 1.1)
    ax_loss.grid(True, alpha=0.25)
    ax_loss.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax_acc.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax_loss.set_title("Recover Period", fontsize=TITLE_FONT_SIZE, pad=10)

    labels = [str(line.get_label()) for line in legend_lines]
    ax_loss.legend(
        legend_lines,
        labels,
        loc="center right",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_period_recovery_history(
    *,
    history_paths: Sequence[Path] | None = None,
    output_path: Path | None = None,
    split: str = "val",
) -> tuple[list[Path], Path]:
    resolved_history_paths = resolve_history_paths(history_paths)
    histories = [
        (history_path, load_history_payload(history_path))
        for history_path in resolved_history_paths
    ]
    resolved_output_path = resolve_output_path(
        resolved_history_paths,
        split,
        output_path,
    )
    plot_history(
        histories=histories,
        output_path=resolved_output_path,
        split=split,
    )
    return resolved_history_paths, resolved_output_path
