from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = Path("figs/fi_fig")
FI_RESULT_CLASS_NAMES = {
    ("fisher_information_utils", "FiResult"),
    ("scripts.fi_data_cal.fisher_information_utils", "FiResult"),
    ("__main__", "FiResult"),
}


@dataclass(frozen=True)
class FiResultRecord:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


class FiResultUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in FI_RESULT_CLASS_NAMES:
            return FiResultRecord
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 calculate_fi_dataset.py 输出的 pkl 文件并绘制 FI 图像。",
    )
    parser.add_argument("input_path", type=Path, help="calculate_fi_dataset.py 输出的 pkl 文件路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"图像输出目录，默认值: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default="fi",
        help="输出图片文件名前缀，默认值: fi",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存图片后额外弹出图窗。",
    )
    return parser.parse_args()


def load_results(input_path: Path) -> list[FiResultRecord]:
    with input_path.open("rb") as file_obj:
        data = FiResultUnpickler(file_obj).load()

    if not isinstance(data, list):
        raise TypeError(f"期望从 {input_path} 中读取到 list，实际得到 {type(data)!r}。")

    normalized_results: list[FiResultRecord] = []
    for item in data:
        if isinstance(item, FiResultRecord):
            normalized_results.append(item)
            continue

        if isinstance(item, dict):
            normalized_results.append(FiResultRecord(**item))
            continue

        raise TypeError(
            "pkl 中的每条记录必须是 FiResultRecord 或可转换为 FiResultRecord 的 dict，"
            f"但读取到了 {type(item)!r}。"
        )

    if not normalized_results:
        raise ValueError(f"{input_path} 中没有可用于绘图的 FI 记录。")

    return normalized_results


def average_by_axis(
    results: list[FiResultRecord],
    axis_name: str,
) -> dict[str, tuple[list[int], list[float]]]:
    grouped_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for result in results:
        axis_value = getattr(result, axis_name)
        if axis_value is None:
            continue
        grouped_values[result.circuit_type][int(axis_value)].append(float(result.fi_value))

    if not grouped_values:
        raise ValueError(f"没有找到可用于横轴 {axis_name!r} 的记录。")

    averaged: dict[str, tuple[list[int], list[float]]] = {}
    for circuit_type, mapping in grouped_values.items():
        sorted_pairs = sorted(mapping.items())
        x_values = [axis_value for axis_value, _ in sorted_pairs]
        y_values = [sum(values) / len(values) for _, values in sorted_pairs]
        averaged[circuit_type] = (x_values, y_values)

    return averaged


def plot_lines(
    averaged_data: dict[str, tuple[list[int], list[float]]],
    xlabel: str,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))

    for circuit_type, (x_values, y_values) in sorted(averaged_data.items()):
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=circuit_type)

    plt.xlabel(xlabel)
    plt.ylabel("FI")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"已保存图像: {output_path}")


def main() -> None:
    args = parse_args()
    results = load_results(args.input_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fi_vs_nqubits = average_by_axis(results, axis_name="nqubit")
    plot_lines(
        fi_vs_nqubits,
        xlabel="nqubits",
        output_path=args.output_dir / f"{args.prefix}_vs_nqubits.png",
        title="FI vs nqubits",
    )

    fi_vs_nlayers = average_by_axis(results, axis_name="nlayer")
    plot_lines(
        fi_vs_nlayers,
        xlabel="nlayer",
        output_path=args.output_dir / f"{args.prefix}_vs_nlayer.png",
        title="FI vs nlayer",
    )

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
