import pytest
import numpy as np
from qiskit import QuantumCircuit
import altqft.nn.process_qc as pc 

def test_make_prob():
    """测试make_prob函数返回的函数是一个正确的概率函数。满足大于0和归一的性质。"""
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    period = 2
    prob = pc.make_prob(qc, period)
    
    N = 2 ** qc.num_qubits
    
    # 验证 sum_col prob(col, shift) = 1 for any shift
    for shift in range(period):
        prob_sum = 0
        for col in range(N):
            p = prob(col, shift)
            assert p >= 0  # 概率必须大于等于0
            prob_sum += p
        assert np.isclose(prob_sum, 1.0)  # 归一性验证
