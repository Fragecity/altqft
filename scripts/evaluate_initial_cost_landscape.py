from __future__ import annotations

import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import pennylane as qml
import torch


EPSILON = 1e-12


def lr_circuit(theta: torch.Tensor, nqubit: int) -> None:
    param_idx = 0
    for wire in range(0, nqubit, 2):
        qml.Hadamard(wires=wire)

    for control in range(0, nqubit, 2):
        for target in range(1, nqubit, 2):
            phase = torch.pi / 2 ** abs(target - control) + theta[param_idx]
            qml.CPhase(phase, wires=[control, target])
            param_idx += 1

    for wire in range(1, nqubit, 2):
        qml.Hadamard(wires=wire)


def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return -torch.sum(p * torch.log(q + EPSILON))


def make_prob(U: torch.Tensor, period: int) -> Callable[[int], torch.Tensor]:
    def prob(x: int) -> torch.Tensor:
        period_elements = U[0::period, x]
        return torch.abs(torch.sum(period_elements)) ** 2 / len(period_elements)

    return prob


def make_cost(nqubit: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def bound_circuit(params: torch.Tensor) -> None:
        lr_circuit(params, nqubit)

    matrix_fn = qml.matrix(bound_circuit, wire_order=range(nqubit))

    def cost_function(params: torch.Tensor) -> torch.Tensor:
        unitary = matrix_fn(params)
        period_start = nqubit
        period_end = max(int(2 ** (nqubit / 4)), nqubit**2)

        distributions: list[torch.Tensor] = []
        for period in range(period_start, period_end):
            prob_fn = make_prob(unitary, period)
            distribution = torch.stack([prob_fn(column) for column in range(2**nqubit)])
            distributions.append(distribution / (torch.sum(distribution) + EPSILON))

        if len(distributions) < 2:
            return torch.tensor(0.0, requires_grad=False)

        ce_values = [
            cross_entropy(distributions[index], distributions[index + 1])
            for index in range(len(distributions) - 1)
        ]
        return torch.min(torch.stack(ce_values))

    return cost_function


def evaluate_and_plot(
    min_q: int = 2,
    max_q: int = 14,
    step: int = 2,
    save_name: str = "init_cost_landscape.png",
) -> None:
    qubit_list = list(range(min_q, max_q + 1, step))
    loss_list: list[float] = []

    print(f"=== 开始评估初始参数下的 Cost (从 {min_q} 到 {max_q} 比特) ===")

    for qubit_count in qubit_list:
        start_time = time.time()
        n_params = len(range(0, qubit_count, 2)) * len(range(1, qubit_count, 2))
        params = torch.zeros(n_params, requires_grad=False)
        cost_fn = make_cost(qubit_count)

        with torch.no_grad():
            loss = cost_fn(params)
            if loss.dim() > 0:
                loss = loss.sum()
            loss_value = loss.item()

        loss_list.append(loss_value)
        elapsed = time.time() - start_time
        print(f"Qubits: {qubit_count:2d} | 初始 Loss: {loss_value:.6f} | 耗时: {elapsed:.2f} 秒")

    plt.figure(figsize=(9, 6))
    plt.plot(
        qubit_list,
        loss_list,
        marker="s",
        markersize=8,
        color="b",
        linewidth=2,
        label="Initial Cost",
    )
    plt.title("Initial Cost Landscape vs Number of Qubits", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Qubits", fontsize=12)
    plt.ylabel("Initial Loss (Smoothed Max-Min CE)", fontsize=12)
    plt.xticks(qubit_list)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ 评估完成！折线图已保存至: {os.path.abspath(save_name)}")


if __name__ == "__main__":
    evaluate_and_plot(min_q=2, max_q=12, step=2, save_name="init_cost_landscape.png")
