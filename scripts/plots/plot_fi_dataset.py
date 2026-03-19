import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

# ================= 配置区 =================
INPUT_FILE = "data/shared/fi_results.pkl"   # 这里填入你的 pkl 数据文件路径
OUTPUT_DIR = "figs/fi_fig"     # 这里填入你想保存图片的文件夹路径
# ==========================================

@dataclass
class FiResultRecord:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: Optional[int] = None

def plot_scatter_and_mean(data_dict, xlabel, output_path):
    plt.figure(figsize=(8, 5))
    
    # 获取 Matplotlib 默认的颜色循环，以确保散点和均值线颜色一致
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (ctype, x_y_dict) in enumerate(sorted(data_dict.items())):
        color = colors[i % len(colors)] # 为当前电路类型分配一种颜色
        
        x_all = []
        y_all = []
        x_mean = []
        y_mean = []
        
        # 按 X 轴排序
        for x_val, y_vals in sorted(x_y_dict.items()):
            # 收集该 X 坐标下的所有散点
            x_all.extend([x_val] * len(y_vals))
            y_all.extend(y_vals)
            
            # 计算该 X 坐标的均值
            x_mean.append(x_val)
            y_mean.append(sum(y_vals) / len(y_vals))
            
        # 1. 绘制所有数据点（散点图），设置透明度防止重叠，并将其添加到图例中
        plt.scatter(x_all, y_all, color=color, alpha=0.5, s=30, label=ctype, zorder=2)
        
        # 2. 绘制均值实线，不传入 label 参数，因此不会出现在图例中
        plt.plot(x_mean, y_mean, color=color, linewidth=2, zorder=1)

    plt.xlabel(xlabel)
    plt.ylabel("Fisher Information (FI)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"已保存图像: {output_path}")

def main():
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载最新版的 pkl 数据
    with input_path.open("rb") as f:
        results = pickle.load(f)

    # 2. 数据分组
    data_nqubit = defaultdict(lambda: defaultdict(list))
    data_nlayer = defaultdict(lambda: defaultdict(list))

    for r in results:
        data_nqubit[r.circuit_type][r.nqubit].append(r.fi_value)
        if r.nlayer is not None:
            data_nlayer[r.circuit_type][r.nlayer].append(r.fi_value)

    # 3. 绘图并保存
    plot_scatter_and_mean(data_nqubit, "Number of Qubits", output_dir / "fi_vs_nqubits.png")
    plot_scatter_and_mean(data_nlayer, "Number of Layers", output_dir / "fi_vs_nlayer.png")

if __name__ == "__main__":
    main()