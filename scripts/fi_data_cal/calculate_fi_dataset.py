from __future__ import annotations

import pickle
import sys
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
)


def build_dataset(config: FiExperimentConfig) -> list[FiResult]:
    return calculate_fi_results(config)


def render_progress_bar(current: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        raise ValueError("total 必须是正整数。")

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio:.0%})"


def print_progress(current: int, total: int, config: FiExperimentConfig) -> None:
    progress_bar = render_progress_bar(current, total)
    layer_text = f", nlayer={config.nlayer}" if config.nlayer is not None else ""
    sys.stdout.write(
        f"\r计算进度 {progress_bar} -> {config.circuit_type}, nqubit={config.nqubit}{layer_text}"
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def load_dataset(input_path: Path) -> list[FiResult]:
    if input_path.exists():
        try:
            with input_path.open("rb") as file_obj:
                data = pickle.load(file_obj)
                if isinstance(data, list):
                    print(f"成功从 {input_path} 加载了 {len(data)} 条历史记录。")
                    return data
        except Exception as e:
            print(f"加载已有数据失败 ({e})，将从头开始记录。")
    else:
        print(f"未找到 {input_path}，将创建新文件。")
    return []


def save_dataset(results: list[FiResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(results, file_obj)


def main() -> None:
    output_path = Path("data/shared/fi_results.pkl")
    existing_results = load_dataset(output_path)

    configs_to_run: list[FiExperimentConfig] = []
    for nqubit in range(9, 12):
        configs_to_run.extend(
            [
                FiExperimentConfig(circuit_type="qft", nqubit=nqubit, repeat=1),
                FiExperimentConfig(circuit_type="ph1", nqubit=nqubit, repeat=1),
                FiExperimentConfig(
                    circuit_type="ph_random",
                    nqubit=nqubit,
                    nlayer=nqubit,
                    repeat=1,
                ),
                FiExperimentConfig(circuit_type="ph_1_random", nqubit=nqubit, repeat=1),
                FiExperimentConfig(
                    circuit_type="ph_random_phase",
                    nqubit=nqubit,
                    nlayer=nqubit,
                    repeat=1,
                ),
            ]
        )

    total_configs = len(configs_to_run)
    new_results: list[FiResult] = []
    for index, config in enumerate(configs_to_run, start=1):
        print_progress(index - 1, total_configs, config)
        current_results = build_dataset(config)
        new_results.extend(current_results)
        print_progress(index, total_configs, config)

    print("本次计算得到的新结果:")
    for result in new_results:
        print(result)

    existing_results.extend(new_results)
    save_dataset(existing_results, output_path)
    print(f"已将总共 {len(existing_results)} 条记录保存至 {output_path}")


if __name__ == "__main__":
    main()
