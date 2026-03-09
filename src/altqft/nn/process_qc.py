import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def make_prob(circuit: QuantumCircuit, period: int) -> Callable[[int, int], float]:
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