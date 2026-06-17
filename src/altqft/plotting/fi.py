from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, cast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

INPUT_FILE = Path("data/shared/fi_results.pkl")
OPTIMIZED_PH1_SUMMARY_FILE = Path("data/shared/ph1_min_fi_summary.json")
HP1_SHARED_SUMMARY_FILE = Path("data/shared/hp1_shared_fi_shift_summary.json")
OUTPUT_DIR = Path("figs/fi_fig")
LABEL_MAP = {
    "ph1": "HP-1, fixed phase",
    "ph1_optimized": "HP-1, optimized phase",
    "HP1_random": "HP-1, random phase",
    "HP1_shared": "HP-1, shared optimized phase",
    "HPrandom": "HP-random, random",
    "HPrandom_phase": "HP-random, random and phase",
}
VARIANCE_BAND_CIRCUITS = {"HPrandom", "HPrandom_phase"}
DATA_EXCLUDED_CIRCUITS = {"HP1_random"}
PLOT_EXCLUDED_CIRCUITS = {"qft"}
CIRCUIT_COLORS = {
    "qft": "black",
    "ph1": "#0081a7",
    "ph1_optimized": "#1d3557",
    "HP1_random": "#00afb9",
    "HPrandom": "#f4a261",
    "HPrandom_phase": "#f07167",
    "HP1_shared": "#1d3557",
}
CIRCUIT_MARKERS = {
    "ph1": "o",
    "ph1_optimized": "s",
    "HP1_random": "^",
    "HPrandom": "D",
    "HPrandom_phase": "p",
    "HP1_shared": "s",
}
LEGEND_ORDER = (
    "ph1",
    "ph1_optimized",
    "HP1_random",
    "HP1_shared",
    "HPrandom",
    "HPrandom_phase",
)
TOP_LAYER_CIRCUITS = {"qft", "ph1"}
BOTTOM_LAYER_CIRCUITS = {"HPrandom_phase"}
FIT_LINEWIDTH = 1.25
FIT_ALPHA = 0.7
FIT_LINESTYLE = "-"
MARKER_SIZE = 28
NQUBITS_LOG_Y_MIN = 0.2
NQUBITS_LOG_Y_MAX = 2.55
NQUBITS_LOG_Y_TICKS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
NQUBITS_DPI = 200


@dataclass(frozen=True)
class FiResultRecord:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


PlotData = DefaultDict[str, DefaultDict[int, list[float]]]


def _format_log_y_tick(value: float, _position: int | None) -> str:
    exponent = round(float(np.log10(value)), 2)
    exponent_text = f"{exponent:g}"
    return rf"$10^{{{exponent_text}}}$"


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


def _log_linear_fit(
    x_values: list[int],
    y_values: list[float],
) -> tuple[list[int], list[float], float] | None:
    positive_points = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values, strict=True)
        if y_value > 0
    ]
    if len(positive_points) < 2:
        return None

    fit_x = np.asarray([point[0] for point in positive_points], dtype=float)
    log_y = np.log([point[1] for point in positive_points])
    slope, intercept = np.polyfit(fit_x, log_y, 1)
    fit_y = np.exp(slope * fit_x + intercept)
    return fit_x.astype(int).tolist(), fit_y.tolist(), float(slope)


def _label_with_fit(
    label: str, fit: tuple[list[int], list[float], float] | None
) -> str:
    if fit is None:
        return label
    return f"{label} (slope={fit[2]:.2f})"


def _plot_fit_line(
    fit: tuple[list[int], list[float], float] | None,
    *,
    color: str,
    zorder: int,
) -> None:
    if fit is None:
        return
    fit_x, fit_y, _ = fit
    plt.plot(
        fit_x,
        fit_y,
        color=color,
        linewidth=FIT_LINEWIDTH,
        linestyle=FIT_LINESTYLE,
        alpha=FIT_ALPHA,
        label="_nolegend_",
        zorder=zorder,
    )


def _plot_mean_points(
    x_values: list[int],
    y_values: list[float],
    *,
    color: str,
    marker: str,
    label: str,
    zorder: int,
) -> None:
    plt.scatter(
        x_values,
        y_values,
        color=color,
        marker=marker,
        s=MARKER_SIZE,
        label=label,
        zorder=zorder,
    )


def plot_fi_vs_nqubits(data_dict: PlotData, output_path: Path) -> None:
    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles: dict[str, Line2D] = {}
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
        marker = CIRCUIT_MARKERS.get(circuit_type, "o")
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
            fit = _log_linear_fit(x_mean, y_mean)
            plt.fill_between(
                x_mean,
                y_lower,
                y_upper,
                color=color,
                alpha=0.18,
                zorder=1,
            )
            _plot_mean_points(
                x_mean,
                y_mean,
                color=color,
                marker=marker,
                label="_nolegend_",
                zorder=line_zorder,
            )
            _plot_fit_line(fit, color=color, zorder=line_zorder + 1)
            legend_handles[circuit_type] = Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=FIT_LINESTYLE,
                linewidth=1.4,
                markersize=5.6,
                label=label,
            )
            continue

        x_mean, y_mean = _mean_points(filtered_x_y_dict)
        fit = _log_linear_fit(x_mean, y_mean)
        _plot_mean_points(
            x_mean,
            y_mean,
            color=color,
            marker=marker,
            label="_nolegend_",
            zorder=line_zorder,
        )
        _plot_fit_line(fit, color=color, zorder=line_zorder + 1)
        legend_handles[circuit_type] = Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=FIT_LINESTYLE,
            linewidth=1.4,
            markersize=5.6,
            label=label,
        )

    plt.yscale("log")
    if positive_values:
        plt.ylim(10**NQUBITS_LOG_Y_MIN, 10**NQUBITS_LOG_Y_MAX)
    axis = plt.gca()
    axis.yaxis.set_major_locator(
        FixedLocator([10**exponent for exponent in NQUBITS_LOG_Y_TICKS])
    )
    axis.yaxis.set_major_formatter(FuncFormatter(_format_log_y_tick))
    plt.xlabel("Qubit Number", fontsize=11, color="#1a1a1a")
    plt.ylabel("Discrete Fisher Information", fontsize=11, color="#1a1a1a")
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ordered_legend_handles = [
        legend_handles[circuit_type]
        for circuit_type in LEGEND_ORDER
        if circuit_type in legend_handles
    ]
    ordered_legend_handles.extend(
        handle
        for circuit_type, handle in legend_handles.items()
        if circuit_type not in LEGEND_ORDER
    )
    plt.legend(
        handles=ordered_legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        labelspacing=0.3,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=NQUBITS_DPI)
    plt.close()


def load_results(input_path: Path) -> list[FiResultRecord]:
    if not input_path.exists():
        return []

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


def load_hp1_shared_results(input_path: Path) -> list[FiResultRecord]:
    if not input_path.exists():
        return []

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records: list[FiResultRecord] = []

    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        nqubit = item.get("nqubit")
        recomputed_min_fi = item.get("min_fi")
        best_epoch = item.get("best_epoch")
        if not isinstance(nqubit, int):
            continue
        if isinstance(recomputed_min_fi, (int, float)):
            fi_value = float(recomputed_min_fi)
        else:
            if not isinstance(best_epoch, dict):
                continue
            min_fi = best_epoch.get("min_fi")
            if isinstance(min_fi, (int, float)):
                fi_value = float(min_fi)
            else:
                continue
        records.append(
            FiResultRecord(
                circuit_type="HP1_shared",
                nqubit=nqubit,
                fi_value=fi_value,
            )
        )
    return records


def filter_results_by_nqubits(
    results: list[FiResultRecord],
    nqubits: set[int] | None,
) -> list[FiResultRecord]:
    if nqubits is None:
        return results
    return [result for result in results if result.nqubit in nqubits]


def filter_excluded_circuits(results: list[FiResultRecord]) -> list[FiResultRecord]:
    return [
        result
        for result in results
        if result.circuit_type not in DATA_EXCLUDED_CIRCUITS
    ]


def group_results_by_qubit(results: list[FiResultRecord]) -> PlotData:
    by_qubit: PlotData = defaultdict(lambda: defaultdict(list))

    for result in results:
        by_qubit[result.circuit_type][result.nqubit].append(result.fi_value)

    return by_qubit


def plot_fi_dataset(
    *,
    input_path: Path = INPUT_FILE,
    optimized_summary_path: Path = OPTIMIZED_PH1_SUMMARY_FILE,
    hp1_shared_summary_path: Path = HP1_SHARED_SUMMARY_FILE,
    output_dir: Path = OUTPUT_DIR,
    nqubit_start: int | None = None,
    nqubit_end: int | None = None,
) -> Path:
    nqubit_filter = resolve_nqubit_filter(nqubit_start, nqubit_end)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = filter_excluded_circuits(
        filter_results_by_nqubits(load_results(input_path), nqubit_filter)
    )
    del optimized_summary_path
    results.extend(
        filter_results_by_nqubits(
            load_hp1_shared_results(hp1_shared_summary_path),
            nqubit_filter,
        )
    )
    output_path = output_dir / "fi_vs_nqubits.svg"
    plot_fi_vs_nqubits(group_results_by_qubit(results), output_path)
    return output_path
