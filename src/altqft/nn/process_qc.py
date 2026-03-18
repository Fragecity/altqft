import numpy as np
from typing import Callable, Iterable, Sequence
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


ProbFunc = Callable[[int, int], float]


def make_prob(circuit: QuantumCircuit, period: int) -> ProbFunc:
    """
    计算给定量子电路的矩阵在指定列上具有shift invariant的概率。
    
    Args:
        circuit (QuantumCircuit): 要检查的量子电路。
        period (int): 期望的周期长度。
        
    Returns:
        Callable[[int, int], float]: 一个函数，接受列索引和shift值，返回矩阵在该列上具有shift invariant的概率。
    """
    U = np.asarray(Operator(circuit).data)
    N = U.shape[0]
    num_k = N // period
    
    def prob(col: int, shift: int) -> float:
        effect_elements = np.array([U[shift + k * period, col] for k in range(num_k)])
        
        sum_val = sum(effect_elements)
        return (np.abs(sum_val) ** 2) / num_k
        
    return prob

def fi(prob1: Sequence[float], prob2: Sequence[float]) -> float:
    """
    计算两个离散概率分布之间的离散 Fisher Information。
    
    Args:
        prob1 (Sequence[float]): 对应 P(x | t) 的概率分布列表或数组。
        prob2 (Sequence[float]): 对应 P(x | t - 1) 的概率分布列表或数组。
        
    Returns:
        float: 计算得到的 Fisher Information 值。
    """
    fisher_info = 0.0
    
    for p_x, p_x_minus_1 in zip(prob1, prob2):
        if p_x > 1e-12:
            fisher_info += ((p_x - p_x_minus_1) ** 2) / p_x
            
    return fisher_info

def min_fi(circuit: QuantumCircuit, period_range: Iterable[int]) -> float:
    """
    计算在给定的多个周期下，相邻周期 (period 与 period+1) 概率分布之间的最小 Fisher Information。
    
    Args:
        circuit (QuantumCircuit): 要计算的量子电路。
        period_range (Iterable[int]): 需要遍历的周期列表或范围。
        
    Returns:
        float: 遍历 period_range 后得到的最小 Fisher Information 值。
    """
    N = 2 ** circuit.num_qubits
    global_min_fi = float('inf')
    shift = 0  
    
    for period in period_range:
        prob_func1 = make_prob(circuit, period)
        prob_func2 = make_prob(circuit, period + 1)
        
        prob1_dist = [prob_func1(col, shift) for col in range(N)]
        prob2_dist = [prob_func2(col, shift) for col in range(N)]
        
        fi_val = fi(prob1_dist, prob2_dist)
        # print(fi_val)
        if fi_val < global_min_fi:
            global_min_fi = fi_val
            
    return global_min_fi

