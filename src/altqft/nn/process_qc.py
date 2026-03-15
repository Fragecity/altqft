import numpy as np
from typing import Callable, Iterable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from dataclasses import dataclass

ProbFunc = Callable[[int, int], float]

@dataclass
class FiData:
    circuit_type: str
    nqubit: int
    nlayer: int
    fi_val: float

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

def fi(sinv_prob: ProbFunc, col: int, period: int) -> float:
    """
    计算指定列 (col) 的离散 Fisher Information。
    """
    fisher_info = 0.0
    for shift in range(period):
        p_x = sinv_prob(col, shift)
        p_x_minus_1 = sinv_prob(col, (shift - 1) % period)
        
        if p_x > 1e-12:
            fisher_info += ((p_x - p_x_minus_1) ** 2) / p_x
            
    return fisher_info

def min_fi(circuit: QuantumCircuit, period_range: Iterable[int]) -> float:
    """
    计算在给定的多个周期下，遍历所有可能初始态（列）后得到的最小 Fisher Information。
    
    Args:
        circuit (QuantumCircuit): 要计算的量子电路。
        period_range (Iterable[int]): 需要遍历的周期列表或范围，例如 [2, 4, 8] 或 range(2, 5)。
        
    Returns:
        float: 所有给定周期和所有初始态中最小的 Fisher Information 值。
    """
    N = 2 ** circuit.num_qubits
    global_min_fi = float('inf')
    
    for period in period_range:

        prob = make_prob(circuit, period)
        

        period_min_fi = min(fi(prob, col, period) for col in range(N))
        
        if period_min_fi < global_min_fi:
            global_min_fi = period_min_fi
            
    return global_min_fi

