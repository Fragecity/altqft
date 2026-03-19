import numpy as np
import random
from qiskit import QuantumCircuit

# 假设这两个文件在同一个目录下，使用相对导入或绝对导入
from altqft.circuits.ph_core import ph_qc, ph_phase 

def _get_num_phases(hlayout: list) -> int:
    """
    内部辅助函数：计算给定 hlayout 构型的电路在 ph_qc 中需要多少个 phase 参数。
    """
    if not hlayout:
        return 0
    max_layer = max(hlayout)
    total_phases = 0
    for i in range(max_layer):
        curr_len = sum(1 for x in hlayout if x == i)
        next_len = sum(1 for x in hlayout if x == i + 1)
        total_phases += curr_len * next_len
    return total_phases


def qft(nqubit: int) -> QuantumCircuit:
    """
    生成标准 QFT 线路（未包含最后的 SWAP 门）
    """
    hlayout = list(range(nqubit))
    phase = np.zeros(int(nqubit*(nqubit-1)/2))
    idx = 0
    for control in range(nqubit):
        for target in range(control + 1, nqubit):
            phase[idx] = np.pi / (2 ** (target - control))
            idx += 1
    return ph_qc(hlayout, phase)


def ph_1(nqubit: int) -> QuantumCircuit:
    """
    生成固定 layout 的电路，模式为 [0, 1, 0, 1, 0, ...]，相位由连接距离固定
    """
    hlayout = [i % 2 for i in range(nqubit)]
    return ph_phase(hlayout)


def ph_random(nqubit: int, nlayer: int) -> QuantumCircuit:
    """
    生成随机 layout 的电路，layout 中的数字随机分布在 0 到 nlayer 之间，相位由连接距离固定
    """
    hlayout = [random.randint(0, nlayer) for _ in range(nqubit)]
    return ph_phase(hlayout)


def ph_1_random(nqubit: int) -> QuantumCircuit:
    """
    生成固定 layout 的电路，模式为 [0, 1, 0, 1, 0, ...]，且所有 phase 参数均为随机生成 (0 ~ 2π)
    """
    hlayout = [i % 2 for i in range(nqubit)]
    num_phases = _get_num_phases(hlayout)
    phases = np.random.uniform(0, 2 * np.pi, num_phases)
    return ph_qc(hlayout, phases)


def ph_random_phase(nqubit: int, nlayer: int) -> QuantumCircuit:
    """
    生成随机 layout 的电路，数字随机分布在 0 到 nlayer 之间，且所有 phase 参数均为随机生成 (0 ~ 2π)
    """
    hlayout = [random.randint(0, nlayer) for _ in range(nqubit)]
    num_phases = _get_num_phases(hlayout)
    phases = np.random.uniform(0, 2 * np.pi, num_phases)
    return ph_qc(hlayout, phases)


if __name__ == "__main__":
    # 测试代码也一并迁移过来，可以直接运行此脚本进行验证
    # qc = ph_qc([0,1,0,1], np.zeros(4))
    # print("ph_qc:")
    # print(qc.draw())
    
    # qc = ph_phase([0,1,0,1])
    # print("ph_phase:")
    # print(qc.draw())
    
    qc = qft(4)
    print("qft:")
    print(qc.draw())

    # # 测试新加的随机相位电路
    qc_1_rand = ph_1_random(4)
    print("ph_1_random (4 qubits):")
    print(qc_1_rand.draw())

    qc_rand_phase = ph_random_phase(4, 2)
    print("ph_random_phase (4 qubits, max 2 layers):")
    print(qc_rand_phase.draw())