from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


DEFAULT_INPUT_PATH = Path(
    "data/shor_ph1_18q_p18-511_foundk_paper.json"
)
DEFAULT_OUTPUT_STEM = Path(
    "figs/recover/shor_ph1_18q_p18-511_foundk_paper"
)
DEFAULT_FORMATS = ("svg", "pdf")
BACKGROUND = "#ffffff"
# Pastel greens for publication, paired with a darker cool gray for contrast.
FOUND_K_COLORS = {
    1: "#9bcfa4",
    2: "#aed8ad",
    3: "#c0dfa5",
    4: "#d0e6b0",
}
NO_SUITABLE_A_COLOR = "#8c96a1"
FAILED_COLOR = "#c98d68"
LABEL_COLOR = "#222222"


@dataclass(frozen=True, slots=True)
class OverviewCell:
    n_value: int
    status: str
    found_k: int | None


@dataclass(frozen=True, slots=True)
class OverviewPayload:
    metadata: dict[str, Any]
    summary: dict[str, Any]
    cells: tuple[OverviewCell, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Shor PH1 semiprime cases by found-k and no-suitable-a status."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSON containing sorted overview cells.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem. Suffixes are controlled by --formats.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FORMATS,
        help="Output formats to write. Defaults to svg pdf.",
    )
    return parser.parse_args(argv)


def load_payload(input_path: Path) -> OverviewPayload:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {input_path}")
    metadata = payload.get("metadata")
    summary = payload.get("summary")
    raw_cells = payload.get("cells")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("cells must be a non-empty JSON array")

    cells: list[OverviewCell] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict):
            raise ValueError("Each cell must be a JSON object")
        n_value = raw_cell.get("N")
        status = raw_cell.get("status")
        found_k = raw_cell.get("found_k")
        if not isinstance(n_value, int):
            raise ValueError("Cell N must be an integer")
        if not isinstance(status, str):
            raise ValueError("Cell status must be a string")
        if found_k is not None and not isinstance(found_k, int):
            raise ValueError("Cell found_k must be null or an integer")
        cells.append(OverviewCell(n_value=n_value, status=status, found_k=found_k))
    return OverviewPayload(
        metadata=metadata,
        summary=summary,
        cells=tuple(sorted(cells, key=lambda cell: cell.n_value)),
    )


def cell_color(cell: OverviewCell) -> str:
    if cell.status == "found":
        if cell.found_k is None:
            raise ValueError(f"Found cell is missing found_k: {cell.n_value}")
        return FOUND_K_COLORS.get(cell.found_k, FOUND_K_COLORS[max(FOUND_K_COLORS)])
    if cell.status == "no_suitable_a":
        return NO_SUITABLE_A_COLOR
    if cell.status == "failed":
        return FAILED_COLOR
    raise ValueError(f"Unknown status {cell.status!r}")


def draw_overview(
    payload: OverviewPayload,
    output_stem: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = int(payload.metadata.get("rows", 42))
    if rows < 1:
        raise ValueError("metadata.rows must be positive")

    cells = payload.cells
    columns = math.ceil(len(cells) / rows)
    fig_width = float(payload.metadata.get("figure_width", 7.2))
    fig_height = float(payload.metadata.get("figure_height", 3.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # Categorical heatmap: each semiprime is one cell, with no grid edges.
    image = [
        [to_rgba(BACKGROUND) for _ in range(columns)]
        for _ in range(rows)
    ]
    for index, cell in enumerate(cells):
        column = index // rows
        row = index % rows
        image[rows - row - 1][column] = to_rgba(cell_color(cell))
    ax.imshow(
        image,
        origin="lower",
        extent=(0, columns, 0, rows),
        interpolation="none",
        aspect="auto",
    )

    found_k_counts = {
        int(k): int(v) for k, v in payload.summary.get("found_k_counts", {}).items()
    }
    no_suitable_a = int(payload.summary.get("no_suitable_a", 0))
    failed = int(payload.summary.get("failed", 0))
    show_header = bool(payload.metadata.get("show_header", True))
    if show_header:
        nqubit = int(payload.metadata.get("nqubit", 18))
        n_start = int(payload.metadata.get("N_start", cells[0].n_value))
        n_stop = int(payload.metadata.get("N_stop", cells[-1].n_value))
        header_parts = [
            f"{nqubit}q semiprimes N={n_start}..{n_stop}",
            *[f"k={k}: {found_k_counts[k]}" for k in sorted(found_k_counts)],
            f"no a: {no_suitable_a}",
            f"failed: {failed}",
            f"rows {rows}",
        ]
        ax.text(
            0,
            rows + 1.35,
            " | ".join(header_parts),
            color=LABEL_COLOR,
            fontsize=float(payload.metadata.get("header_font_size", 8.4)),
            ha="left",
            va="bottom",
        )

    tick_columns = [0, columns // 5, 2 * columns // 5, 3 * columns // 5, 4 * columns // 5, columns - 1]
    tick_columns = sorted(set(tick_columns))
    ax.set_xticks([column + 0.5 for column in tick_columns])
    ax.set_xticklabels(
        [str(cells[min(column * rows, len(cells) - 1)].n_value) for column in tick_columns],
        fontsize=float(payload.metadata.get("tick_font_size", 6.8)),
        color="#333333",
    )
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([])

    legend_items = [
        Patch(facecolor=FOUND_K_COLORS[k], edgecolor=FOUND_K_COLORS[k], label=f"k={k} ({found_k_counts[k]})")
        for k in sorted(found_k_counts)
    ]
    legend_items.append(
        Patch(
            facecolor=NO_SUITABLE_A_COLOR,
            edgecolor=NO_SUITABLE_A_COLOR,
            label=f"no suitable a ({no_suitable_a})",
        )
    )
    if failed:
        legend_items.append(
            Patch(facecolor=FAILED_COLOR, edgecolor=FAILED_COLOR, label=f"failed ({failed})")
        )
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, float(payload.metadata.get("legend_anchor_y", -0.16))),
        ncol=len(legend_items),
        frameon=False,
        fontsize=float(payload.metadata.get("legend_font_size", 7.4)),
        handlelength=1.4,
        columnspacing=2.6,
    )

    ax.set_xlim(0, columns)
    y_margin = 2.7 if show_header else float(payload.metadata.get("top_margin", 0.45))
    ax.set_ylim(0, rows + y_margin)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [output_stem.with_suffix(f".{output_format.lstrip('.')}") for output_format in formats]
    for output_path in output_paths:
        fig.savefig(output_path, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return output_paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = load_payload(args.input)
    output_paths = draw_overview(
        payload=payload,
        output_stem=args.output_stem,
        formats=[str(output_format) for output_format in args.formats],
    )
    print(f"input_path={args.input}")
    print(f"cells={len(payload.cells)}")
    print("output_paths=" + ",".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
