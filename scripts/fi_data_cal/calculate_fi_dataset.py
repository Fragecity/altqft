from __future__ import annotations

import pickle
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
)


def build_dataset(config: FiExperimentConfig) -> list[FiResult]:
    """接收一个配置，然后跑这个配置并返回结果"""
    return calculate_fi_results(config)


def load_dataset(input_path: Path) -> list[FiResult]:
    """尝试加载已有的 pickle 数据。"""
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
    
    # 1. 先尝试读取历史数据
    existing_results = load_dataset(output_path)
    
    # 2. 构造我们需要跑的任务配置
    configs_to_run: list[FiExperimentConfig] = []
    
    # 测试 4 ~ 10 比特的情况
    for nqubit in range(4, 11):
        # QFT
        configs_to_run.append(
            FiExperimentConfig(circuit_type="qft", nqubit=nqubit, repeat=1)
        )
        
        # PH1
        configs_to_run.append(
            FiExperimentConfig(circuit_type="ph1", nqubit=nqubit, repeat=1)
        )
        
        # PH_random: 这里的 nlayer 根据你的需要传入，暂时设为与 nqubit 一致
        configs_to_run.append(
            FiExperimentConfig(circuit_type="ph_random", nqubit=nqubit, nlayer=nqubit, repeat=1)
        )

    # 3. 循环调用 build_dataset 跑任务
    new_results: list[FiResult] = []
    for config in configs_to_run:
        current_results = build_dataset(config)
        new_results.extend(current_results)
    
    print("本次计算得到的新结果:")
    for result in new_results:
        print(result)
        
    # 4. 将新结果追加到历史数据中
    existing_results.extend(new_results)
    
    # 5. 保存合并后的完整数据集
    save_dataset(existing_results, output_path)
    print(f"已将总共 {len(existing_results)} 条记录保存至 {output_path}")


if __name__ == "__main__":
    main()