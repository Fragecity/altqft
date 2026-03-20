from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from qiskit import QuantumCircuit

from altqft.circuits.ph_generators import ph_1_hlayout, ph_1_parametrized


def _single_qubit_gate(gate: Tensor, qubit: int, nqubit: int) -> Tensor:
    """将单比特门扩展到 nqubit 维 Hilbert 空间。"""
    factors: list[Tensor] = []
    identity = torch.eye(2, dtype=gate.dtype, device=gate.device)

    for idx in reversed(range(nqubit)):
        factors.append(gate if idx == qubit else identity)

    full_gate = factors[0]
    for factor in factors[1:]:
        full_gate = torch.kron(full_gate, factor)
    return full_gate


def _controlled_phase_gate(theta: Tensor, control: int, target: int, nqubit: int) -> Tensor:
    """构造受控相位门的完整矩阵。"""
    size = 2**nqubit
    gate = torch.eye(size, dtype=torch.complex64, device=theta.device)
    phase = torch.exp(1j * theta.to(torch.complex64))

    for basis in range(size):
        control_bit = (basis >> control) & 1
        target_bit = (basis >> target) & 1
        if control_bit == 1 and target_bit == 1:
            gate[basis, basis] = phase

    return gate


def _probability_distribution(unitary: Tensor, period: int, shift: int = 0) -> Tensor:
    """根据线路矩阵计算固定 period 下所有列的 shift-invariant 概率分布。"""
    dimension = unitary.shape[0]
    num_k = dimension // period
    row_indices = shift + torch.arange(num_k, device=unitary.device) * period
    selected_rows = unitary.index_select(0, row_indices)
    amplitudes = selected_rows.sum(dim=0)
    raw_prob = amplitudes.abs().pow(2) / float(num_k)
    return raw_prob / raw_prob.sum().clamp_min(1e-12)


def fisher_information(prob1: Tensor, prob2: Tensor, eps: float = 1e-8) -> Tensor:
    """Torch 版本的离散 Fisher information。"""
    denominator = prob1.clamp_min(eps)
    return ((prob1 - prob2).pow(2) / denominator).sum()


class PH1MinFIModel(nn.Module):
    """使用 ph_1 固定 hlayout、以最小 Fisher information 为目标的可训练模型。"""

    def __init__(self, nqubit: int, init_phases: Sequence[float] | None = None) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.hlayout = ph_1_hlayout(nqubit)
        self.phase_count = self._count_required_phases()

        if init_phases is None:
            initial_tensor = 2 * math.pi * torch.rand(self.phase_count, dtype=torch.float32)
        else:
            if len(init_phases) != self.phase_count:
                raise ValueError(
                    f"PH1MinFIModel 需要 {self.phase_count} 个初始化参数，实际收到 {len(init_phases)} 个。"
                )
            initial_tensor = torch.tensor(init_phases, dtype=torch.float32)

        self.phases = nn.Parameter(initial_tensor)
        self.register_buffer(
            "hadamard",
            torch.tensor(
                [[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64
            )
            / math.sqrt(2.0),
        )

    def _count_required_phases(self) -> int:
        total_phases = 0
        remaining_qubits = self.nqubit
        max_layer = max(self.hlayout)

        for layer in range(max_layer + 1):
            current_len = sum(1 for value in self.hlayout if value == layer)
            remaining_qubits -= current_len
            total_phases += current_len * remaining_qubits

        return total_phases

    def build_unitary(self) -> Tensor:
        """按 ph_qc 的门顺序构造当前参数对应的线路矩阵。"""
        dimension = 2**self.nqubit
        unitary = torch.eye(dimension, dtype=torch.complex64, device=self.phases.device)
        phase_index = 0
        rest_qubits = set(range(self.nqubit))

        for layer in range(max(self.hlayout)):
            current_layer = [idx for idx, value in enumerate(self.hlayout) if value == layer]
            rest_qubits -= set(current_layer)
            targets = sorted(rest_qubits)

            for control in current_layer:
                hadamard_gate = _single_qubit_gate(self.hadamard, control, self.nqubit)
                unitary = hadamard_gate @ unitary
                for target in targets:
                    cp_gate = _controlled_phase_gate(
                        self.phases[phase_index], control, target, self.nqubit
                    )
                    unitary = cp_gate @ unitary
                    phase_index += 1

        last_layer = [idx for idx, value in enumerate(self.hlayout) if value == max(self.hlayout)]
        for qubit in last_layer:
            hadamard_gate = _single_qubit_gate(self.hadamard, qubit, self.nqubit)
            unitary = hadamard_gate @ unitary

        return unitary

    def min_fi(self, period_range: Iterable[int]) -> Tensor:
        """计算当前参数下的最小 Fisher information。"""
        periods = list(period_range)
        if not periods:
            raise ValueError("period_range 不能为空。")

        unitary = self.build_unitary()
        dimension = unitary.shape[0]
        min_value: Tensor | None = None

        for period in periods:
            if period <= 0 or period >= dimension:
                raise ValueError(f"period={period} 不在有效范围内，必须满足 1 <= period < {dimension}。")
            if period + 1 >= dimension:
                raise ValueError(
                    f"period+1={period + 1} 超出维度范围，当前线路维度为 {dimension}。"
                )

            prob1 = _probability_distribution(unitary, period)
            prob2 = _probability_distribution(unitary, period + 1)
            fi_value = fisher_information(prob1, prob2)
            min_value = fi_value if min_value is None else torch.minimum(min_value, fi_value)

        if min_value is None:
            raise RuntimeError("未能计算 min_fi。")
        return min_value

    def forward(self, period_range: Iterable[int]) -> Tensor:
        return self.min_fi(period_range)

    def export_phases(self) -> list[float]:
        """导出当前训练后的 phase 参数。"""
        return self.phases.detach().cpu().tolist()

    def current_circuit(self) -> QuantumCircuit:
        """将当前参数导出为 Qiskit QuantumCircuit，便于保存或后续分析。"""
        return ph_1_parametrized(self.nqubit, self.export_phases())
