from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, cast

import matplotlib.pyplot as plt

INPUT_FILE = Path("data/shared/fi_results.pkl")
OUTPUT_DIR = Path("figs/fi_fig")
FI_DATA_DIR = Path(__file__).resolve().parent.parent / "fi_data_cal"


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _scatter_points(x_y_dict: DefaultDict[int, list[float]]) -> tuple[list[int], list[float]]:
    x_all: list[int] = []
    y_all: list[float] = []

    for x_value, y_values in sorted(x_y_dict.items()):
        x_all.extend([x_value] * len(y_values))
        y_all.extend(y_values)

    return x_all, y_all


def _mean_points(x_y_dict: DefaultDict[int, list[float]]) -> tuple[list[int], list[float]]:
    x_mean: list[int] = []
    y_mean: list[float] = []

    for x_value, y_values in sorted(x_y_dict.items()):
        x_mean.append(x_value)
        y_mean.append(_mean(y_values))

    return x_mean, y_mean


def plot_scatter_and_mean(data_dict: PlotData, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (circuit_type, x_y_dict) in enumerate(sorted(data_dict.items())):
        color = colors[index % len(colors)]
        x_all, y_all = _scatter_points(x_y_dict)
        x_mean, y_mean = _mean_points(x_y_dict)
        plt.scatter(x_all, y_all, color=color, alpha=0.5, s=30, label=circuit_type, zorder=2)
        plt.plot(x_mean, y_mean, color=color, linewidth=2, zorder=1)

    plt.xlabel(xlabel)
    plt.ylabel("Fisher Information")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_results(input_path: Path) -> list[FiResultRecord]:
    with input_path.open("rb") as file_obj:
        return cast(list[FiResultRecord], pickle.load(file_obj))


def group_results(results: list[FiResultRecord]) -> tuple[PlotData, PlotData]:
    by_qubit: PlotData = defaultdict(lambda: defaultdict(list))
    by_layer: PlotData = defaultdict(lambda: defaultdict(list))

    for result in results:
        by_qubit[result.circuit_type][result.nqubit].append(result.fi_value)
        if result.nlayer is not None:
            by_layer[result.circuit_type][result.nlayer].append(result.fi_value)

    return by_qubit, by_layer


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_pickle_dependencies()

    results = load_results(INPUT_FILE)
    by_qubit, by_layer = group_results(results)
    plot_scatter_and_mean(by_qubit, "Number of Qubits", OUTPUT_DIR / "fi_vs_nqubits.png")
    plot_scatter_and_mean(by_layer, "Number of Layers", OUTPUT_DIR / "fi_vs_nlayer.png")


if __name__ == "__main__":
    main()
