import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def is_shift_invariant(circuit: QuantumCircuit, col: int, period: int) -> bool:
    """
    检查给定的量子电路的矩阵在指定列上是否具有shift invariant。
    
    Args:
        circuit (QuantumCircuit): 要检查的量子电路。
        col (int): 要检查的列索引。
        period (int): 期望的周期长度。

    Returns:
        bool: 如果矩阵在指定列上具有shift invariant，则返回True；否则返回False。
    
    Example:
        >>> is_shift_invariant(circuit, col=0, period=2)
        True  # 如果矩阵在第0列上具有周期为2的shift invariant
    """
    prob = make_prob(circuit, period)
    p = prob(col, 0)
    for shift in range(1, period):
        if not np.isclose(prob(col, shift), p):
            # print(f"Shift {shift} has different probability: {prob(col, shift)} != {p}")
            return False
    return True

def make_prob(circuit: QuantumCircuit, period: int) -> Callable[[int, int], float]:
    """
    计算给定量子电路的矩阵在指定列上具有shift invariant的概率。
    
    Args:
        circuit (QuantumCircuit): 要检查的量子电路。
        period (int): 期望的周期长度。
        
    Returns:
        Callable[[int, int], float]: 一个函数，接受列索引和shift值，返回矩阵在该列上具有shift invariant的概率。
    """
    # 获取电路的幺正矩阵
    U = np.asarray(Operator(circuit).data)
    N = U.shape[0]
    num_k = N // period
    
    def prob(col: int, shift: int) -> float:
        # 按照公式计算: abs( sum_k U_{shift+k*period, col} )^2 / num_k
        sum_val = sum(U[shift + k * period, col] for k in range(num_k))
        return (np.abs(sum_val) ** 2) / num_k
        
    return prob