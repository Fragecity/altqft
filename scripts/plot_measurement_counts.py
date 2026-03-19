from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorSampler

from altqft.circuits.ph import qft


N_QUBITS = 8
SHOTS = 20_000
MIN_DECIMAL_STATE = 1
OUTPUT_PATH = Path("figs/fi_fig/qft_measurement_counts.png")


def sample_counts(nqubits: int, shots: int) -> dict[str, int]:
    circuit = qft(nqubits)
    circuit.measure_all()
    sampler = StatevectorSampler()
    job = sampler.run([circuit], shots=shots)
    result = job.result()
    return result[0].data.meas.get_counts()


def filter_and_convert_counts(counts: dict[str, int], min_decimal_state: int) -> dict[int, int]:
    decimal_counts = {
        int(bitstring, 2): count
        for bitstring, count in counts.items()
        if int(bitstring, 2) >= min_decimal_state
    }
    return dict(sorted(decimal_counts.items()))


def plot_counts(counts: dict[int, int], nqubits: int, shots: int, save_path: Path) -> None:
    if not counts:
        raise ValueError("过滤后的 counts 为空，无法绘图。")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = list(counts.keys())
    y_values = list(counts.values())

    plt.figure(figsize=(16, 6))
    plt.bar(x_values, y_values, color="royalblue", width=2.0)
    plt.title(
        f"QFT Measurement Counts vs Decimal State\n(n={nqubits}, shots={shots})",
        fontsize=14,
    )
    plt.xlabel("Measured State (Decimal)", fontsize=12)
    plt.ylabel(f"Counts (out of {shots} shots)", fontsize=12)
    plt.xlim(max(MIN_DECIMAL_STATE - 1, 0), 2**nqubits)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 绘图成功！柱状图已保存至: {save_path}")


def main() -> None:
    print(f"正在模拟 {N_QUBITS} Qubits 的 QFT 测量分布 (shots={SHOTS})...")
    counts = sample_counts(N_QUBITS, SHOTS)
    filtered_counts = filter_and_convert_counts(counts, MIN_DECIMAL_STATE)
    plot_counts(filtered_counts, N_QUBITS, SHOTS, OUTPUT_PATH)


if __name__ == "__main__":
    main()
