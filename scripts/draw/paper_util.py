import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from altqft.circuits.ph import qft
from altqft.nn.process_qc import make_prob

def calculate_distributions(nqubit: int, period: int, shift: int = 0) -> Tuple[List[float], List[float]]:
    N = 2 ** nqubit
    circuit = qft(nqubit)
    
    prob_func1 = make_prob(circuit, period)
    prob_func2 = make_prob(circuit, period + 1)
    
    prob1_dist = [prob_func1(col, shift) for col in range(N)]
    prob2_dist = [prob_func2(col, shift) for col in range(N)]
    
    return prob1_dist, prob2_dist

def plot_distributions(dist1: List[float], dist2: List[float], nqubit: int, save_path: str) -> None:
    N = len(dist1)
    x_axis = range(N)
    
    # 设置画布大小
    plt.figure(figsize=(12, 6))
    
    color1 = '#8ecae6'
    color2 = '#219ebc'
    
    plt.plot(x_axis, dist1, color=color1, linestyle='-', linewidth=1.5)
    plt.fill_between(x_axis, dist1, color=color1, alpha=0.5)
    
    plt.plot(x_axis, dist2, color=color2, linestyle='-', linewidth=1.5)
    plt.fill_between(x_axis, dist2, color=color2, alpha=0.5)
    
    plt.xlabel('computational basis')
    plt.ylabel('counts')
    plt.title(f'QFT Probability Distributions (qubit num={nqubit})')
    
    # 隐藏刻度
    plt.xticks([])
    plt.yticks([])
    
    # 确保网格关闭
    plt.grid(False)
    
    # --- 新增逻辑：淡化外部边框 ---
    ax = plt.gca()  # 获取当前坐标轴
    for spine in ax.spines.values():
        spine.set_alpha(0.3)  # 设置边框透明度为 0.3 (淡化)
    # --------------------------------

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # --- 修改保存逻辑：移除白边 ---
    # bbox_inches='tight' 自动裁剪空白，pad_inches=0 确保无填充
    plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.show()

def main() -> None:
    nqubit = 10
    period = 17
    shift = 0
    save_path = 'figs/fi_fig/qft_prob_dist.svg'
    
    prob1_dist, prob2_dist = calculate_distributions(nqubit, period, shift)
    plot_distributions(prob1_dist, prob2_dist, nqubit, save_path)

if __name__ == "__main__":
    main()