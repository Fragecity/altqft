import random
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from altqft.circuits.ph_core import ph_phase, ph_qc


def _get_num_phases(hlayout: list[int]) -> int:
    """计算给定 hlayout 构型在 ph_qc 中需要的 phase 参数数量。"""
    total_phases = 0
    remaining_qubits = len(hlayout)

    for layer in range(max(hlayout) + 1):
        curr_len = sum(1 for x in hlayout if x == layer)
        remaining_qubits -= curr_len
        total_phases += curr_len * remaining_qubits

    return total_phases


def ph_1_hlayout(nqubit: int) -> list[int]:
    """返回 ph_1 使用的固定 hlayout，模式为 [0, 1, 0, 1, ...]。"""
    return [i % 2 for i in range(nqubit)]


def qft(nqubit: int) -> QuantumCircuit:
    """生成标准 QFT 线路（未包含最后的 SWAP 门）。"""
    hlayout = list(range(nqubit))
    phase = np.zeros(int(nqubit * (nqubit - 1) / 2))
    idx = 0
    for control in range(nqubit):
        for target in range(control + 1, nqubit):
            phase[idx] = np.pi / (2 ** (target - control))
            idx += 1
    return ph_qc(hlayout, phase)


def ph_1(nqubit: int) -> QuantumCircuit:
    """生成固定 layout 的电路，模式为 [0, 1, 0, 1, ...]。"""
    return ph_phase(ph_1_hlayout(nqubit))


def ph_1_parametrized(nqubit: int, phases: Sequence[float] | np.ndarray) -> QuantumCircuit:
    """使用 ph_1 的 hlayout 生成电路，但 phase 参数由调用者显式提供。"""
    hlayout = ph_1_hlayout(nqubit)
    expected_num_phases = _get_num_phases(hlayout)
    phase_array = np.asarray(phases, dtype=float)

    if phase_array.shape != (expected_num_phases,):
        raise ValueError(
            f"ph_1_parametrized 需要 {expected_num_phases} 个 phase 参数，"
            f"实际收到 {phase_array.size} 个。"
        )

    return ph_qc(hlayout, phase_array)


def ph_random(nqubit: int, nlayer: int) -> QuantumCircuit:
    """生成随机 layout 的电路，相位由连接距离固定。"""
    hlayout = [random.randint(0, nlayer) for _ in range(nqubit)]
    return ph_phase(hlayout)


def ph_1_random(nqubit: int) -> QuantumCircuit:
    """生成固定 layout 电路，并为所有 phase 参数随机赋值。"""
    hlayout = ph_1_hlayout(nqubit)
    num_phases = _get_num_phases(hlayout)
    phases = np.random.uniform(0, 2 * np.pi, num_phases)
    return ph_qc(hlayout, phases)


def ph_random_phase(nqubit: int, nlayer: int) -> QuantumCircuit:
    """生成随机 layout 电路，并为所有 phase 参数随机赋值。"""
    hlayout = [random.randint(0, nlayer) for _ in range(nqubit)]
    num_phases = _get_num_phases(hlayout)
    phases = np.random.uniform(0, 2 * np.pi, num_phases)
    return ph_qc(hlayout, phases)


if __name__ == "__main__":
    qc = qft(4)
    print("qft:")
    print(qc.draw())

    qc_1_rand = ph_1_random(4)
    print("ph_1_random (4 qubits):")
    print(qc_1_rand.draw())

    qc_1_param = ph_1_parametrized(4, np.linspace(0.1, 0.6, 6))
    print("ph_1_parametrized (4 qubits):")
    print(qc_1_param.draw())

    qc_rand_phase = ph_random_phase(4, 2)
    print("ph_random_phase (4 qubits, max 2 layers):")
    print(qc_rand_phase.draw())
