import math
from unittest.mock import patch
from qiskit import QuantumCircuit
from altqft.nn.process_qc import *

# ----------------- 测试用例 1: 均匀分布 -----------------
def mock_prob_uniform(col: int, shift: int) -> float:
    return 0.25

def test_uniform_distribution():
    period = 4
    col = 0
    
    # 1. 测试 fi 函数本身
    fi_value = fi(mock_prob_uniform, col, period)
    print(f"均匀分布 FI (预期 0.0): {fi_value}")
    assert math.isclose(fi_value, 0.0, abs_tol=1e-9)

    # 2. 测试 min_fi 函数
    # patch 拦截 make_prob，让它直接返回我们的 mock_prob_uniform
    with patch('altqft.nn.process_qc.make_prob', return_value=mock_prob_uniform):
        # 随便创建一个 1 qubit 的电路，只是为了提供 N=2 的维度
        dummy_circuit = QuantumCircuit(1) 
        min_fi_value = min_fi(dummy_circuit, period_range=[4])
        print(f"均匀分布 min_fi (预期 0.0): {min_fi_value}")
        assert math.isclose(min_fi_value, 0.0, abs_tol=1e-9)

# ----------------- 测试用例 2: 非均匀分布 -----------------
def mock_prob_alternating(col: int, shift: int) -> float:
    if col == 0:
        return 0.8 if shift % 2 == 0 else 0.2
    else:
        return 0.6 if shift % 2 == 0 else 0.4

def test_alternating_distribution():
    period = 2
    
    # 1. 测试 fi 函数 (针对 col=0 和 col=1 分别测试)
    fi_col_0 = fi(mock_prob_alternating, 0, period)
    fi_col_1 = fi(mock_prob_alternating, 1, period)
    
    print(f"非均匀分布 col=0 FI (预期 2.25): {fi_col_0}")
    assert math.isclose(fi_col_0, 2.25, rel_tol=1e-9)
    
    print(f"非均匀分布 col=1 FI (预期 ~0.16667): {fi_col_1}")
    assert math.isclose(fi_col_1, 1/6, rel_tol=1e-9)

    # 2. 测试 min_fi 函数 (应该挑出两列中的最小值 1/6)
    with patch('altqft.nn.process_qc.make_prob', return_value=mock_prob_alternating):
        dummy_circuit = QuantumCircuit(1) # 1 qubit 意味着会遍历 col=0 和 col=1
        min_fi_value = min_fi(dummy_circuit, period_range=[2])
        print(f"非均匀分布 min_fi (预期 ~0.16667): {min_fi_value}")
        assert math.isclose(min_fi_value, 1/6, rel_tol=1e-9)

if __name__ == "__main__":
    print("--- 运行测试 ---")
    test_uniform_distribution()
    print("-" * 20)
    test_alternating_distribution()
    print("所有测试通过！✅")