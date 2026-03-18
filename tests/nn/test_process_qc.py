import pytest
import math
from unittest.mock import patch
from qiskit import QuantumCircuit
from altqft.nn.process_qc import fi, min_fi

# ----------------- 测试用例 1: fi 函数基础测试 -----------------
def test_fi_basic():
    """
    测试离散 Fisher Information 的纯数学计算逻辑。
    公式: FI = sum((p1 - p2)^2 / p1)
    """
    # 场景 A: 两个概率分布完全相同，FI 应该为 0
    p1_same = [0.5, 0.5]
    p2_same = [0.5, 0.5]
    fi_zero = fi(p1_same, p2_same)
    print(f"相同分布 FI (预期 0.0): {fi_zero}")
    assert math.isclose(fi_zero, 0.0, abs_tol=1e-9)

    # 场景 B: 两个不同的概率分布
    # p1 = [0.8, 0.2], p2 = [0.6, 0.4]
    # FI = (0.8 - 0.6)^2 / 0.8 + (0.2 - 0.4)^2 / 0.2 
    #    = 0.04 / 0.8 + 0.04 / 0.2 = 0.05 + 0.20 = 0.25
    p1_diff = [0.8, 0.2]
    p2_diff = [0.6, 0.4]
    fi_value = fi(p1_diff, p2_diff)
    print(f"不同分布 FI (预期 0.25): {fi_value}")
    assert math.isclose(fi_value, 0.25, rel_tol=1e-9)


# ----------------- 测试用例 2: min_fi 综合测试 -----------------

def mock_make_prob_dynamic(circuit: QuantumCircuit, period: int):
    """
    一个伪造的 make_prob 工厂函数。
    它根据传入的 period 返回不同的概率分布函数，用来模拟随周期演化的量子态。
    """
    def prob(col: int, shift: int) -> float:
        # 为了测试，我们假定是 1 qubit (N=2，即 col=0 或 1)，并且不考虑 shift
        if period == 2:
            return 0.8 if col == 0 else 0.2  # 分布 [0.8, 0.2]
        elif period == 3:
            return 0.6 if col == 0 else 0.4  # 分布 [0.6, 0.4]
        elif period == 4:
            return 0.5 if col == 0 else 0.5  # 分布 [0.5, 0.5]
        else:
            return 0.5 
    return prob

def test_min_fi_dynamic():
    """
    测试 min_fi 是否正确遍历了 period_range，并比较了相邻周期的概率分布。
    """
    # 我们计划传入 period_range = [2, 3]
    # 
    # 内部执行逻辑:
    # 1. period = 2 时:
    #    比较 period=2 ([0.8, 0.2]) 和 period=3 ([0.6, 0.4]) 
    #    FI = 0.25 (从 test_fi_basic 已知)
    #
    # 2. period = 3 时:
    #    比较 period=3 ([0.6, 0.4]) 和 period=4 ([0.5, 0.5])
    #    FI = (0.6-0.5)^2/0.6 + (0.4-0.5)^2/0.4 
    #       = 0.01/0.6 + 0.01/0.4 = 1/60 + 1/40 = 5/120 = 1/24 ≈ 0.0416667
    #
    # 最小 FI 应为 min(0.25, 1/24) = 1/24
    
    # 拦截 make_prob，用 side_effect 让它能动态处理不同的入参
    with patch('altqft.nn.process_qc.make_prob', side_effect=mock_make_prob_dynamic):
        dummy_circuit = QuantumCircuit(1) # 1 qubit，对应 N=2 
        
        min_fi_value = min_fi(dummy_circuit, period_range=[2, 3])
        expected_min_fi = 1 / 24
        
        print(f"动态周期 min_fi (预期 {expected_min_fi:.6f}): {min_fi_value:.6f}")
        assert math.isclose(min_fi_value, expected_min_fi, rel_tol=1e-9)

