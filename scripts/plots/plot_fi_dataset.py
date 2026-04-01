from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import DefaultDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"

INPUT_FILE = Path("data/shared/fi_results.pkl")
OPTIMIZED_PH1_SUMMARY_FILE = Path("data/shared/ph1_min_fi_summary.json")
OUTPUT_DIR = Path("figs/fi_fig")
FI_DATA_DIR = Path(__file__).resolve().parent.parent / "fi_data_cal"
LABEL_MAP = {"ph1_optimized": "optimized 1ph", "ph_1_random": "ph1_random"}
VARIANCE_BAND_CIRCUITS = {"ph_1_random", "ph_random", "ph_random_phase"}
PLOT_EXCLUDED_CIRCUITS = {"qft"}
CIRCUIT_COLORS = {
    "qft": "black",
    "ph1": "#0081a7",
    "ph1_optimized": "#1d3557",
    "ph_1_random": "#00afb9",
    "ph_random": "#fed9b7",
    "ph_random_phase": "#f07167",
}
TOP_LAYER_CIRCUITS = {"qft", "ph1"}
BOTTOM_LAYER_CIRCUITS = {"ph_random_phase"}
LAYER_PLOT_EXCLUDED_CIRCUITS = {"ph_random"}
BASE_LINEWIDTH = 1.7
NLAYER_LEGEND_FONT_SCALE = 1.15
NQUBITS_SVG_SIZE_PT = (280, 300)
NQUBITS_FIGSIZE = (NQUBITS_SVG_SIZE_PT[0] / 72, NQUBITS_SVG_SIZE_PT[1] / 72)
NQUBITS_DPI = 200
NLAYER_SVG_SIZE_PT = (220, 210)
NLAYER_FIGSIZE = (NLAYER_SVG_SIZE_PT[0] / 72, NLAYER_SVG_SIZE_PT[1] / 72)
PLOT_DPI = 200
Y_AXIS_LABEL = "Minimum Discrete Fisher Info"
Y_LABEL_FONT_SCALE = 0.9
VIOLIN_WHISKER_COLOR = "#9c6a67"
VIOLIN_MEDIAN_COLOR = "#7f1d1d"
VIOLIN_WIDTH = 0.38


@dataclass(frozen=True)
class FiResultRecord:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


PlotData = DefaultDict[str, DefaultDict[int, list[float]]]


def ensure_pickle_dependencies() -> None:
    fi_data_dir = str(FI_DATA_DIR)
    if fi_data_dir not in sys.path:
        sys.path.insert(0, fi_data_dir)


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


def resolve_nqubit_filter(start: int | None, end: int | None) -> set[int] | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("nqubit start and end must be provided together")
    if start < 2:
        raise ValueError("nqubit start must be at least 2")
    if end < start:
        raise ValueError("nqubit end must be greater than or equal to start")
    return set(range(start, end + 1))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    return float(np.var(values))


def _scaled_fontsize(value: str | float | int, scale: float) -> float:
    return FontProperties(size=value).get_size_in_points() * scale


def _scatter_points(
    x_y_dict: DefaultDict[int, list[float]],
) -> tuple[list[int], list[float]]:
    x_all: list[int] = []
    y_all: list[float] = []

    for x_value, y_values in sorted(x_y_dict.items()):
        x_all.extend([x_value] * len(y_values))
        y_all.extend(y_values)

    return x_all, y_all


def _mean_points(
    x_y_dict: DefaultDict[int, list[float]],
) -> tuple[list[int], list[float]]:
    x_mean: list[int] = []
    y_mean: list[float] = []

    for x_value, y_values in sorted(x_y_dict.items()):
        x_mean.append(x_value)
        y_mean.append(_mean(y_values))

    return x_mean, y_mean


def _mean_and_variance_points(
    x_y_dict: DefaultDict[int, list[float]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    x_values: list[int] = []
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for x_value, y_values in sorted(x_y_dict.items()):
        mean_value = _mean(y_values)
        variance_value = _variance(y_values)
        min_value = min(y_values)
        max_value = max(y_values)
        x_values.append(x_value)
        means.append(mean_value)
        lowers.append(max(mean_value - variance_value, min_value, 1e-12))
        uppers.append(min(mean_value + variance_value, max_value))

    return x_values, means, lowers, uppers


def _positive_only(
    x_y_dict: DefaultDict[int, list[float]],
) -> DefaultDict[int, list[float]]:
    filtered: DefaultDict[int, list[float]] = defaultdict(list)

    for x_value, y_values in x_y_dict.items():
        positive_values = [value for value in y_values if value > 0]
        if positive_values:
            filtered[x_value].extend(positive_values)

    return filtered


def plot_scatter_and_mean(data_dict: PlotData, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=NQUBITS_FIGSIZE)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    positive_values = [
        value
        for circuit_type, x_y_dict in data_dict.items()
        if circuit_type not in PLOT_EXCLUDED_CIRCUITS
        for y_values in x_y_dict.values()
        for value in y_values
        if value > 0
    ]

    for index, (circuit_type, x_y_dict) in enumerate(sorted(data_dict.items())):
        if circuit_type in PLOT_EXCLUDED_CIRCUITS:
            continue
        color = CIRCUIT_COLORS.get(circuit_type, colors[index % len(colors)])
        label = LABEL_MAP.get(circuit_type, circuit_type)
        if circuit_type in TOP_LAYER_CIRCUITS:
            line_zorder = 5
        elif circuit_type in BOTTOM_LAYER_CIRCUITS:
            line_zorder = 0
        else:
            line_zorder = 2

        filtered_x_y_dict = _positive_only(x_y_dict)
        if not filtered_x_y_dict:
            continue

        if circuit_type in VARIANCE_BAND_CIRCUITS:
            x_mean, y_mean, y_lower, y_upper = _mean_and_variance_points(
                filtered_x_y_dict
            )
            plt.fill_between(
                x_mean, y_lower, y_upper, color=color, alpha=0.18, zorder=1
            )
            plt.plot(
                x_mean,
                y_mean,
                color=color,
                linewidth=BASE_LINEWIDTH,
                label=label,
                zorder=line_zorder,
            )
            continue

        x_mean, y_mean = _mean_points(filtered_x_y_dict)
        plt.plot(
            x_mean,
            y_mean,
            color=color,
            linewidth=BASE_LINEWIDTH,
            label=label,
            zorder=line_zorder,
        )

    if positive_values:
        y_min = min(positive_values)
        plt.ylim(max(0.0, y_min * 0.95), 50.0)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        labelspacing=0.3,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=NQUBITS_DPI)
    plt.close()


def _adjacent_values(
    sorted_values: np.ndarray, q1: float, q3: float
) -> tuple[float, float]:
    iqr = q3 - q1
    upper_adjacent_value = q3 + 1.5 * iqr
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, sorted_values[-1])

    lower_adjacent_value = q1 - 1.5 * iqr
    lower_adjacent_value = np.clip(lower_adjacent_value, sorted_values[0], q1)
    return float(lower_adjacent_value), float(upper_adjacent_value)


def plot_layer_violin(data_dict: PlotData, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=NLAYER_FIGSIZE)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    circuit_types = sorted(
        circuit_type
        for circuit_type in data_dict
        if circuit_type not in LAYER_PLOT_EXCLUDED_CIRCUITS
        and circuit_type not in PLOT_EXCLUDED_CIRCUITS
    )
    all_layers = sorted(
        {
            layer
            for circuit_type, x_y_dict in data_dict.items()
            if circuit_type not in LAYER_PLOT_EXCLUDED_CIRCUITS
            and circuit_type not in PLOT_EXCLUDED_CIRCUITS
            for layer in x_y_dict
        }
    )

    if not circuit_types or not all_layers:
        plt.close(fig)
        return

    offsets = (
        np.linspace(-0.15, 0.15, len(circuit_types))
        if len(circuit_types) > 1
        else np.array([0.0])
    )
    legend_handles: list[Line2D] = []

    for index, circuit_type in enumerate(circuit_types):
        color = CIRCUIT_COLORS.get(circuit_type, colors[index % len(colors)])
        x_y_dict = data_dict[circuit_type]
        layers = [layer for layer in all_layers if layer in x_y_dict]
        if not layers:
            continue

        values = [x_y_dict[layer] for layer in layers]
        positions = np.array(layers, dtype=float) + offsets[index]
        violin = ax.violinplot(
            values,
            positions=positions,
            widths=VIOLIN_WIDTH,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in cast(list[PolyCollection], violin["bodies"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)

        quartile1 = np.array([np.percentile(sample, 25) for sample in values])
        medians = np.array([np.percentile(sample, 50) for sample in values])
        quartile3 = np.array([np.percentile(sample, 75) for sample in values])

        whiskers = np.array(
            [
                _adjacent_values(np.sort(np.asarray(sample, dtype=float)), q1, q3)
                for sample, q1, q3 in zip(values, quartile1, quartile3)
            ]
        )
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        ax.scatter(
            positions,
            medians,
            marker="o",
            color=VIOLIN_MEDIAN_COLOR,
            s=30,
            zorder=3,
        )
        ax.vlines(
            positions,
            whiskers_min,
            whiskers_max,
            color=VIOLIN_WHISKER_COLOR,
            linestyle="-",
            lw=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=8,
                alpha=0.45,
                label=LABEL_MAP.get(circuit_type, circuit_type),
            )
        )

    ax.set_xticks(all_layers)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        frameon=False,
        fontsize=_scaled_fontsize(
            plt.rcParams["legend.fontsize"], NLAYER_LEGEND_FONT_SCALE
        ),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def load_results(input_path: Path) -> list[FiResultRecord]:
    with input_path.open("rb") as file_obj:
        return cast(list[FiResultRecord], pickle.load(file_obj))


def load_optimized_ph1_results(input_path: Path) -> list[FiResultRecord]:
    if not input_path.exists():
        return []

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records: list[FiResultRecord] = []

    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        nqubit = item.get("nqubit")
        best_epoch = item.get("best_epoch")
        if not isinstance(nqubit, int) or not isinstance(best_epoch, dict):
            continue
        min_fi = best_epoch.get("min_fi")
        if isinstance(min_fi, (int, float)):
            fi_value = float(min_fi)
        else:
            loss = best_epoch.get("loss")
            if not isinstance(loss, (int, float)):
                continue
            fi_value = float(-loss)
        records.append(
            FiResultRecord(
                circuit_type="ph1_optimized",
                nqubit=nqubit,
                fi_value=fi_value,
            )
        )

    existing_nqubits = {record.nqubit for record in records}
    history_key_pattern = re.compile(r"^ph1_min_fi_(\d+)q_history$")
    for key, value in payload.items():
        match = history_key_pattern.match(key)
        if match is None or not isinstance(value, list):
            continue

        nqubit = int(match.group(1))
        if nqubit in existing_nqubits:
            continue

        best_fi: float | None = None
        for history_item in value:
            if not isinstance(history_item, dict):
                continue
            min_fi = history_item.get("min_fi")
            if isinstance(min_fi, (int, float)):
                candidate = float(min_fi)
            else:
                loss = history_item.get("loss")
                if not isinstance(loss, (int, float)):
                    continue
                candidate = float(-loss)
            if best_fi is None or candidate > best_fi:
                best_fi = candidate

        if best_fi is None:
            continue

        records.append(
            FiResultRecord(
                circuit_type="ph1_optimized",
                nqubit=nqubit,
                fi_value=best_fi,
            )
        )
        existing_nqubits.add(nqubit)

    return records


def filter_results_by_nqubits(
    results: list[FiResultRecord],
    nqubits: set[int] | None,
) -> list[FiResultRecord]:
    if nqubits is None:
        return results
    return [result for result in results if result.nqubit in nqubits]


def group_results(results: list[FiResultRecord]) -> tuple[PlotData, PlotData]:
    by_qubit: PlotData = defaultdict(lambda: defaultdict(list))
    by_layer: PlotData = defaultdict(lambda: defaultdict(list))

    for result in results:
        by_qubit[result.circuit_type][result.nqubit].append(result.fi_value)
        if result.nlayer is not None:
            by_layer[result.circuit_type][result.nlayer].append(result.fi_value)

    return by_qubit, by_layer


def main() -> None:
    args = parse_args()
    nqubit_filter = resolve_nqubit_filter(args.nqubit_start, args.nqubit_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_pickle_dependencies()

    results = filter_results_by_nqubits(load_results(args.input), nqubit_filter)
    results.extend(
        filter_results_by_nqubits(
            load_optimized_ph1_results(args.optimized_summary),
            nqubit_filter,
        )
    )
    by_qubit, by_layer = group_results(results)
    plot_scatter_and_mean(
        by_qubit, "Number of Qubits", args.output_dir / "fi_vs_nqubits.svg"
    )
    plot_layer_violin(by_layer, args.output_dir / "fi_vs_nlayer.svg")


if __name__ == "__main__":
    main()
