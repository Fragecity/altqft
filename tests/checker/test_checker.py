import pytest
import numpy as np
from qiskit import QuantumCircuit
# 修正了导入名，使其与 shift_inv.py 中的函数名匹配
from altqft.checker.shift_inv import is_shift_invariant, make_prob

def test_make_prob():
    """测试make_prob函数返回的函数是一个正确的概率函数。满足大于0和归一的性质。"""
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    period = 2
    prob = make_prob(qc, period)
    
    N = 2 ** qc.num_qubits
    
    # 验证 sum_col prob(col, shift) = 1 for any shift
    for shift in range(period):
        prob_sum = 0
        for col in range(N):
            p = prob(col, shift)
            assert p >= 0  # 概率必须大于等于0
            prob_sum += p
        assert np.isclose(prob_sum, 1.0)  # 归一性验证

def test_check_shift_invariance_transversal_h():
    """测试check_shift_invariance函数在H^tensor n的电路上返回True。"""
    n = 2
    qc = QuantumCircuit(n)
    qc.h(range(n))
    
    # H^tensor n 电路的矩阵在任何列上都具有一致的振幅分布，必然是 Shift Invariant 的
    assert is_shift_invariant(qc, col=0, period=2) is True

def test_check_shift_invariance_non_sinv():
    """测试check_shift_invariance函数在不具有shift invariant的电路上返回False。"""
    qc = QuantumCircuit(2)
    # 不对电路做任何操作，等效于单位矩阵 Identity。
    # 它的第一列是 [1, 0, 0, 0]^T。
    # 对于 period=2，shift=0 的概率为 1/2，shift=1 的概率为 0，因此必定返回 False。
    assert is_shift_invariant(qc, col=0, period=2) is False