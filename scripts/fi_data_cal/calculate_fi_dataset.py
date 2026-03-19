from __future__ import annotations

import pickle
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
)


def build_dataset(config: FiExperimentConfig) -> list[FiResult]:
    return calculate_fi_results(config)


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
    for nqubit in range(4, 11):
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

    new_results: list[FiResult] = []
    for config in configs_to_run:
        current_results = build_dataset(config)
        new_results.extend(current_results)

    print("本次计算得到的新结果:")
    for result in new_results:
        print(result)

    existing_results.extend(new_results)
    save_dataset(existing_results, output_path)
    print(f"已将总共 {len(existing_results)} 条记录保存至 {output_path}")


if __name__ == "__main__":
    main()
