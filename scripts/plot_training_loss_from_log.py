from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


QUBIT_PATTERN = re.compile(r"开始训练\s+(\d+)\s+比特线路")
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d\.\-]+)")


def parse_training_log(file_path: Path) -> dict[int, dict[str, list[float]]]:
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")

    parsed: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"epochs": [], "losses": []}
    )
    current_qubits: int | None = None

    with file_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            qubit_match = QUBIT_PATTERN.search(line)
            if qubit_match:
                current_qubits = int(qubit_match.group(1))
                continue

            if current_qubits is None:
                continue

            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                parsed[current_qubits]["epochs"].append(int(epoch_match.group(1)))
                parsed[current_qubits]["losses"].append(float(epoch_match.group(2)))

    if not parsed:
        raise ValueError("未能在文件中解析到任何训练数据，请检查文件内容格式。")

    return dict(parsed)


def plot_loss_landscape(file_path: str, save_path: str = "training_loss_landscape.png") -> None:
    data = parse_training_log(Path(file_path))
    plt.figure(figsize=(10, 6))

    for qubits in sorted(data):
        epochs = data[qubits]["epochs"]
        losses = data[qubits]["losses"]
        if epochs and losses:
            plt.plot(epochs, losses, marker="o", markersize=3, linewidth=1.5, label=f"{qubits} Qubits")

    plt.title("Training Loss vs Epochs for Quantum Circuits", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Circuit Scale", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 绘图成功！图表已保存至: {save_path}")


if __name__ == "__main__":
    plot_loss_landscape("data/training_landscape.txt")
