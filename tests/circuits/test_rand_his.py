import numpy as np
from qiskit.quantum_info import Operator
# 导入重构后的函数
from altqft.circuits.rand_his import rand_h_skeleton, rand_cp_block, build_rand_h_circuit

def test_rand_h_skeleton():
    """测试每一个比特上仅有一个Hadamard门，且每个比特上的Hadamard门位置在[0, nlayers)范围内。"""
    nlayers = 2
    h_skl = rand_h_skeleton(nqubits=3, nlayers=nlayers)
    
    assert len(h_skl) == 3
    assert all(0 <= pos < nlayers for pos in h_skl)

def test_rand_cp_block():
    """测试cp_block作用在 computational basis 上之后的结果仍然是 computational basis，只是相位改变了。"""
    nqubits = 3
    cp_block = rand_cp_block(nqubits=nqubits)
    
    # 提取电路矩阵
    op = Operator(cp_block).data
    diag_elements = np.diagonal(op)
    
    # 验证矩阵为对角阵（非对角线元素必须全为0）
    assert np.allclose(np.diag(diag_elements), op), "CP block should be a diagonal matrix"
    
    # 验证所有对角线元素的绝对值均为 1.0 （模长不变，只改变了相位）
    assert np.allclose(np.abs(diag_elements), 1.0), "Magnitudes of diagonal elements must be exactly 1.0"

def test_build_rand_h_circuit():
    """测试构建的线路的矩阵形式。矩阵元是 1/sqrt(N) e^(i*theta) 的形式。"""
    nqubits = 3
    qc, _ = build_rand_h_circuit(nqubits=nqubits, nlayers=2)
    
    # 提取完整电路矩阵
    op = Operator(qc).data
    
    # 预期的振幅大小：N = 2^nqubits
    expected_magnitude = 1.0 / np.sqrt(2 ** nqubits)
    
    # 验证最终矩阵中每个元素的模长都等于 1 / sqrt(N)
    assert np.allclose(np.abs(op), expected_magnitude), f"All matrix elements must have magnitude {expected_magnitude}"