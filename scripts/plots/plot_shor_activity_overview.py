from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch


DEFAULT_LOG_PATH = Path("outputs/evaluate_shor_ph1_semiprimes_exact_shiftce_pool10_hold1_11_1800_covered.log")
DEFAULT_OUTPUT_STEM = Path("figs/illus/shor_activity_overview_11_1800")
DEFAULT_ROWS = 7
CELL_STEP = 1.08
OUTER_SIZE = 0.9
OUTER_FACE = "#f6f8fa"
OUTER_EDGE = "#d0d7de"
SOLVED_EDGE_WIDTH = 0.0
UNSOLVED_FACE = "#e2e8f0"
UNSOLVED_EDGE = "#cbd5e1"
UNCOVERED_FACE = "#f8fafc"
UNCOVERED_EDGE = "#94a3b8"
BACKGROUND = "#ffffff"
TITLE_COLOR = "#0f172a"
SUBTITLE_COLOR = "#475569"
LABEL_COLOR = "#64748b"
K_COLORS: dict[int, str] = {
    1: "#013a63",
    2: "#2a6f97",
    3: "#89c2d9",
    4: "#89c2d9",
    5: "#89c2d9",
    6: "#89c2d9",
    7: "#89c2d9",
    8: "#89c2d9",
    9: "#89c2d9",
    10: "#89c2d9",
}


@dataclass(frozen=True, slots=True)
class CaseRecord:
    N: int
    expected_factors: tuple[int, int]
    a: int
    found_k: int | None
    top_periods: tuple[int, ...]
    top1_success: bool
    topk_success: bool

    @property
    def minimal_k(self) -> int | None:
        if self.found_k is not None:
            return self.found_k
        for index, period in enumerate(self.top_periods, start=1):
            if factors_from_order(self.a, self.N, period) == self.expected_factors:
                return index
        return None


@dataclass(frozen=True, slots=True)
class GridCell:
    N: int
    status: str
    minimal_k: int | None = None
    top1_success: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    start: int
    stop: int
    case_records: tuple[CaseRecord, ...]
    uncovered_case_ids: tuple[int, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a GitHub-activity-like semiprime overview for the PH1 + DeepSet Shor replacement. "
            "Solved covered cases are colored by the minimum k needed to recover the correct factors."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Evaluation log produced by evaluate_shor_ph1_semiprimes.py.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem. The script writes both .png and .svg.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="How many rows to use in the activity grid.",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=None,
        help="Optional lower bound for the plotted semiprime range.",
    )
    parser.add_argument(
        "--range-stop",
        type=int,
        default=None,
        help="Optional upper bound for the plotted semiprime range.",
    )
    parser.add_argument(
        "--covered-only",
        action="store_true",
        help="Only draw covered cases and hide uncovered semiprimes.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Hide the chart title and subtitle.",
    )
    parser.add_argument(
        "--hide-summary",
        action="store_true",
        help="Hide the bottom summary statistics line.",
    )
    parser.add_argument(
        "--cell-font-size",
        type=float,
        default=6.8,
        help="Font size for the numbers inside the squares.",
    )
    return parser.parse_args(argv)


def factors_from_order(a: int, modulus: int, order: int) -> tuple[int, int] | None:
    if order % 2 != 0:
        return None
    half_power = pow(a, order // 2, modulus)
    if half_power in (1, modulus - 1):
        return None
    left = math.gcd(half_power - 1, modulus)
    right = math.gcd(half_power + 1, modulus)
    if 1 < left < modulus and 1 < right < modulus:
        return tuple(sorted((left, right)))
    return None


def prime_factors_with_multiplicity(n: int) -> list[int]:
    factors: list[int] = []
    remainder = n
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor += 1 if divisor == 2 else 2
    if remainder > 1:
        factors.append(remainder)
    return factors


def semiprime_numbers(start: int, stop: int) -> list[int]:
    values: list[int] = []
    for number in range(start, stop + 1):
        if len(prime_factors_with_multiplicity(number)) == 2:
            values.append(number)
    return values


def parse_boolean(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Expected boolean text, got {text!r}")


def parse_key_value_fields(prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in prefix.split()[1:]:
        key, value = token.split("=", maxsplit=1)
        fields[key] = value
    return fields


def parse_case_line(line: str) -> CaseRecord:
    head, separator, tail = line.partition(" found=")
    if not separator:
        raise ValueError(f"Malformed case line: {line}")
    field_map = parse_key_value_fields(head)

    _, separator, top_periods_text = tail.partition(" top_periods=")
    if not separator:
        raise ValueError(f"Missing top_periods field: {line}")
    top_periods = ast.literal_eval(top_periods_text)
    if not isinstance(top_periods, list) or not all(isinstance(value, int) for value in top_periods):
        raise ValueError(f"Invalid top_periods payload: {line}")

    expected_left, expected_right = field_map["expected"].split("x", maxsplit=1)
    found_k_text = field_map.get("found_k", "None")
    found_k = None if found_k_text == "None" else int(found_k_text)
    return CaseRecord(
        N=int(field_map["N"]),
        expected_factors=tuple(sorted((int(expected_left), int(expected_right)))),
        a=int(field_map["a"]),
        found_k=found_k,
        top_periods=tuple(top_periods),
        top1_success=parse_boolean(field_map["top1_success"]),
        topk_success=parse_boolean(field_map["topk_success"]),
    )


def load_dataset(log_path: Path) -> EvaluationDataset:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    config_fields: dict[str, str] | None = None
    uncovered_case_ids: tuple[int, ...] = ()
    case_records: list[CaseRecord] = []

    for line in lines:
        if line.startswith("config "):
            config_fields = parse_key_value_fields(line)
        elif line.startswith("uncovered_case_ids="):
            payload = ast.literal_eval(line.split("=", maxsplit=1)[1])
            if not isinstance(payload, list) or not all(isinstance(value, int) for value in payload):
                raise ValueError(f"Invalid uncovered_case_ids payload in {log_path}")
            uncovered_case_ids = tuple(payload)
        elif line.startswith("case "):
            case_records.append(parse_case_line(line))

    if config_fields is None:
        raise ValueError(f"Missing config line in {log_path}")
    if not case_records:
        raise ValueError(f"No case rows found in {log_path}")

    start = int(config_fields["start"])
    stop = int(config_fields["requested_stop"])
    return EvaluationDataset(
        start=start,
        stop=stop,
        case_records=tuple(sorted(case_records, key=lambda record: record.N)),
        uncovered_case_ids=tuple(sorted(uncovered_case_ids)),
    )


def build_grid_cells(
    dataset: EvaluationDataset,
    *,
    range_start: int | None = None,
    range_stop: int | None = None,
    covered_only: bool = False,
) -> list[GridCell]:
    case_map = {record.N: record for record in dataset.case_records}
    uncovered_set = set(dataset.uncovered_case_ids)
    cells: list[GridCell] = []
    missing: list[int] = []
    visible_start = dataset.start if range_start is None else max(dataset.start, range_start)
    visible_stop = dataset.stop if range_stop is None else min(dataset.stop, range_stop)

    if visible_start > visible_stop:
        raise ValueError("Requested plot range is empty after intersecting with the dataset range")

    if covered_only:
        for number in sorted(case_map):
            if not (visible_start <= number <= visible_stop):
                continue
            record = case_map[number]
            status = "solved" if record.topk_success and record.minimal_k is not None else "unsolved"
            cells.append(
                GridCell(
                    N=number,
                    status=status,
                    minimal_k=record.minimal_k,
                    top1_success=record.top1_success,
                )
            )
        return cells

    for number in semiprime_numbers(visible_start, visible_stop):
        if number in case_map:
            record = case_map[number]
            status = "solved" if record.topk_success and record.minimal_k is not None else "unsolved"
            cells.append(
                GridCell(
                    N=number,
                    status=status,
                    minimal_k=record.minimal_k,
                    top1_success=record.top1_success,
                )
            )
        elif number in uncovered_set:
            cells.append(GridCell(N=number, status="uncovered"))
        else:
            missing.append(number)

    if missing:
        raise ValueError(f"Log does not account for semiprime cases: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return cells


def square_color(status: str, k: int | None) -> str:
    if status == "solved":
        if k is None:
            raise ValueError("Solved cell is missing k")
        return K_COLORS.get(k, K_COLORS[max(K_COLORS)])
    if status == "unsolved":
        return UNSOLVED_FACE
    if status == "uncovered":
        return UNCOVERED_FACE
    raise ValueError(f"Unknown status {status}")


def square_scale(status: str, k: int | None) -> float:
    if status == "solved":
        if k is None:
            raise ValueError("Solved cell is missing k")
        return max(0.5, 0.94 - 0.045 * (k - 1))
    if status == "unsolved":
        return 0.84
    if status == "uncovered":
        return 0.76
    raise ValueError(f"Unknown status {status}")


def text_color(face_color: str, status: str) -> str:
    if status != "solved":
        return "#1f2937"
    red, green, blue = to_rgb(face_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#000000" if luminance >= 0.55 else "#ffffff"


def draw_round_square(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    size: float,
    face_color: str,
    edge_color: str,
    linewidth: float,
    rounding: float,
    linestyle: str = "solid",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            size,
            size,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor=face_color,
            linestyle=linestyle,
        )
    )


def build_summary(dataset: EvaluationDataset, cells: Sequence[GridCell]) -> tuple[int, int, int, int, int]:
    total_semiprimes = len(cells)
    covered_cases = sum(1 for cell in cells if cell.status != "uncovered")
    uncovered_cases = sum(1 for cell in cells if cell.status == "uncovered")
    top1_successes = sum(1 for cell in cells if cell.status == "solved" and cell.top1_success)
    topk_successes = sum(1 for cell in cells if cell.status == "solved")
    return total_semiprimes, covered_cases, uncovered_cases, top1_successes, topk_successes


def visible_counts(
    dataset: EvaluationDataset,
    *,
    visible_start: int,
    visible_stop: int,
) -> tuple[int, int, int]:
    semiprime_total = len(semiprime_numbers(visible_start, visible_stop))
    covered_total = sum(1 for record in dataset.case_records if visible_start <= record.N <= visible_stop)
    uncovered_total = sum(1 for number in dataset.uncovered_case_ids if visible_start <= number <= visible_stop)
    return semiprime_total, covered_total, uncovered_total


def draw_activity_overview(
    dataset: EvaluationDataset,
    cells: Sequence[GridCell],
    output_stem: Path,
    rows: int,
    *,
    covered_only: bool = False,
    hide_title: bool = False,
    hide_summary: bool = False,
    cell_font_size: float = 6.8,
) -> list[Path]:
    if rows < 1:
        raise ValueError("rows must be positive")

    columns = math.ceil(len(cells) / rows)
    width = columns * CELL_STEP
    height = rows * CELL_STEP
    total_semiprimes, covered_cases, uncovered_cases, top1_successes, topk_successes = build_summary(dataset, cells)
    top1_rate = 100.0 * top1_successes / covered_cases if covered_cases else 0.0
    topk_rate = 100.0 * topk_successes / covered_cases if covered_cases else 0.0
    visible_start = cells[0].N
    visible_stop = cells[-1].N
    visible_semiprimes, visible_covered, visible_uncovered = visible_counts(
        dataset,
        visible_start=visible_start,
        visible_stop=visible_stop,
    )

    fig_width = max(16.0, columns * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 9.2), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for index, cell in enumerate(cells):
        column = index // rows
        row = index % rows
        cell_origin_x = column * CELL_STEP
        cell_origin_y = row * CELL_STEP
        outer_x = cell_origin_x + (CELL_STEP - OUTER_SIZE) / 2.0
        outer_y = cell_origin_y + (CELL_STEP - OUTER_SIZE) / 2.0
        draw_round_square(
            ax,
            x=outer_x,
            y=outer_y,
            size=OUTER_SIZE,
            face_color=OUTER_FACE,
            edge_color=OUTER_EDGE,
            linewidth=0.8,
            rounding=0.12,
        )

        inner_size = OUTER_SIZE * square_scale(cell.status, cell.minimal_k)
        inner_x = cell_origin_x + (CELL_STEP - inner_size) / 2.0
        inner_y = cell_origin_y + (CELL_STEP - inner_size) / 2.0
        face_color = square_color(cell.status, cell.minimal_k)
        edge_color = face_color
        edge_width = SOLVED_EDGE_WIDTH
        linestyle = "solid"
        if cell.status == "unsolved":
            edge_color = UNSOLVED_EDGE
            edge_width = 0.8
        elif cell.status == "uncovered":
            edge_color = UNCOVERED_EDGE
            edge_width = 1.0
            linestyle = (0, (2.0, 1.4))

        draw_round_square(
            ax,
            x=inner_x,
            y=inner_y,
            size=inner_size,
            face_color=face_color,
            edge_color=edge_color,
            linewidth=edge_width,
            rounding=0.12,
            linestyle=linestyle,
        )

        ax.text(
            cell_origin_x + CELL_STEP / 2.0,
            cell_origin_y + CELL_STEP / 2.0,
            str(cell.N),
            ha="center",
            va="center",
            color=text_color(face_color, cell.status),
            fontsize=cell_font_size,
            fontweight="bold",
            family="DejaVu Sans Mono",
        )

    header_stride = max(1, math.ceil(columns / 12))
    for column in range(0, columns, header_stride):
        index = column * rows
        if index >= len(cells):
            continue
        ax.text(
            column * CELL_STEP + CELL_STEP / 2.0,
            -0.26,
            str(cells[index].N),
            ha="center",
            va="center",
            color=LABEL_COLOR,
            fontsize=9.5,
        )

    if not hide_title:
        ax.text(
            0.0,
            -1.42,
            f"Alternative Shor Activity Overview ({visible_start} <= N <= {visible_stop})",
            ha="left",
            va="center",
            color=TITLE_COLOR,
            fontsize=17,
            fontweight="bold",
        )
        ax.text(
            0.0,
            -0.96,
            (
                "Numbers are semiprimes N. "
                + (
                    "This is a covered-only view; every cell shown entered the factorization evaluation."
                    if covered_only
                    else "Solved covered cases use your k colors; dashed cells are uncovered by the candidate-period constraint."
                )
            ),
            ha="left",
            va="center",
            color=SUBTITLE_COLOR,
            fontsize=9.8,
        )
    if not hide_summary:
        ax.text(
            0.0,
            height + 0.34,
            (
                f"Semiprimes {visible_semiprimes}    shown {total_semiprimes}    covered {visible_covered}    uncovered {visible_uncovered}    "
                f"top1 {top1_successes}/{covered_cases} = {top1_rate:.1f}%    top10 {topk_successes}/{covered_cases} = {topk_rate:.1f}%"
            ),
            ha="left",
            va="center",
            color=SUBTITLE_COLOR,
            fontsize=10.4,
        )

    present_k_values = sorted({cell.minimal_k for cell in cells if cell.status == "solved" and cell.minimal_k is not None})
    legend_items: list[tuple[str, str, int | None]] = [(f"k={k}", "solved", k) for k in present_k_values]
    if not covered_only:
        legend_items.append(("covered but unsolved", "unsolved", None))
        legend_items.append(("uncovered", "uncovered", None))
    legend_columns = min(5, len(legend_items))
    legend_rows = math.ceil(len(legend_items) / legend_columns)
    legend_start_y = height + (0.18 if hide_summary else 0.74)
    legend_x_step = 3.08
    legend_y_step = 0.62

    for item_index, (label, status, k_value) in enumerate(legend_items):
        legend_column = item_index % legend_columns
        legend_row = item_index // legend_columns
        legend_x = legend_column * legend_x_step
        legend_y = legend_start_y + legend_row * legend_y_step
        face_color = square_color(status, k_value)
        size = 0.38 if status != "solved" else 0.42 * square_scale(status, k_value)
        edge_color = face_color
        edge_width = SOLVED_EDGE_WIDTH
        linestyle = "solid"
        if status == "unsolved":
            edge_color = UNSOLVED_EDGE
            edge_width = 0.8
        elif status == "uncovered":
            edge_color = UNCOVERED_EDGE
            edge_width = 1.0
            linestyle = (0, (2.0, 1.4))
        draw_round_square(
            ax,
            x=legend_x + (0.5 - size) / 2.0,
            y=legend_y + (0.5 - size) / 2.0,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            linewidth=edge_width,
            rounding=0.08,
            linestyle=linestyle,
        )
        ax.text(
            legend_x + 0.58,
            legend_y + 0.24,
            label,
            ha="left",
            va="center",
            color=LABEL_COLOR,
            fontsize=9.6,
        )

    legend_width = legend_columns * legend_x_step
    content_width = max(width, legend_width)
    extra_bottom = 0.55 + legend_rows * legend_y_step

    ax.set_xlim(-0.2, content_width + 0.2)
    top_margin = -0.55 if hide_title else -1.82
    ax.set_ylim(height + extra_bottom, top_margin)
    ax.set_aspect("equal")
    ax.axis("off")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")]
    for output_path in outputs:
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_dataset(Path(args.log))
    cells = build_grid_cells(
        dataset,
        range_start=args.range_start,
        range_stop=args.range_stop,
        covered_only=bool(args.covered_only),
    )
    output_paths = draw_activity_overview(
        dataset=dataset,
        cells=cells,
        output_stem=Path(args.output_stem),
        rows=int(args.rows),
        covered_only=bool(args.covered_only),
        hide_title=bool(args.hide_title),
        hide_summary=bool(args.hide_summary),
        cell_font_size=float(args.cell_font_size),
    )
    print(f"log_path={Path(args.log)}")
    print(f"cells={len(cells)}")
    print(f"covered_cases={sum(1 for cell in cells if cell.status != 'uncovered')}")
    print(f"uncovered_cases={sum(1 for cell in cells if cell.status == 'uncovered')}")
    print("outputs=" + ",".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
