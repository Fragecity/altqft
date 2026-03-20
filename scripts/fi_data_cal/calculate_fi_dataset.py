# calculate_fi_dataset.py
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
    # 现在内部会根据 config.repeat 自动生成新的随机电路并计算
    return calculate_fi_results(config)


def render_progress_bar(current: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return f"[{'#' * width}]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio:.0%})"


def print_progress(current: int, total: int, config: FiExperimentConfig) -> None:
    progress_bar = render_progress_bar(current, total)
    layer_text = f", nlayer={config.nlayer}" if config.nlayer is not None else ""
    sys.stdout.write(
        f"\r计算进度 {progress_bar} -> {config.circuit_type}, nqubit={config.nqubit}{layer_text} (repeat={config.repeat})   "
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

    # 实验超参数
    NQUBIT_RANGE = range(4, 11)  # 从 4 到 8 比特
    SAMPLES = 64                # 随机线路的采样次数

    configs_to_run: list[FiExperimentConfig] = []
    
    for nqubit in NQUBIT_RANGE:
        # 1. 确定性线路及无须深度对比的随机线路
        configs_to_run.extend([
            FiExperimentConfig(circuit_type="qft", nqubit=nqubit, repeat=1),
            FiExperimentConfig(circuit_type="ph1", nqubit=nqubit, repeat=1),
            # 布局固定、仅相位随机的线路，跑多次求均值即可，不需要遍历 nlayer
            FiExperimentConfig(circuit_type="ph_1_random", nqubit=nqubit, repeat=SAMPLES),
        ])

        # 2. 需要对比不同深度的随机线路 (nlayer 的范围必须严格小于 nqubit)
        for nlayer in range(1, nqubit):
            configs_to_run.extend([
                FiExperimentConfig(
                    circuit_type="ph_random",
                    nqubit=nqubit,
                    nlayer=nlayer,
                    repeat=SAMPLES,
                ),
                FiExperimentConfig(
                    circuit_type="ph_random_phase",
                    nqubit=nqubit,
                    nlayer=nlayer,
                    repeat=SAMPLES,
                ),
            ])

    total_configs = len(configs_to_run)
    new_results: list[FiResult] = []
    
    print(f"\n计划执行 {total_configs} 组配置 (包含多次采样)...")
    for index, config in enumerate(configs_to_run, start=1):
        print_progress(index - 1, total_configs, config)
        current_results = build_dataset(config)
        new_results.extend(current_results)
        print_progress(index, total_configs, config)

    print("\n完成！部分新结果采样:")
    # 仅打印前5条避免刷屏
    for result in new_results[:5]:
        print(result)
    print("...")

    existing_results.extend(new_results)
    save_dataset(existing_results, output_path)
    print(f"已将总共 {len(existing_results)} 条记录保存至 {output_path}")


if __name__ == "__main__":
    main()