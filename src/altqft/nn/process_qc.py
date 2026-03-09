import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

HsProb = Callable[[int, int], float]

def make_prob(circuit: QuantumCircuit, period: int) -> HsProb:
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

def _get_N(prob: Callable) -> int:
    col = 0
    while True:
        try:
            prob(col, 0)
            col += 1
        except IndexError:
            return col

def discrete_fisher_info(prob: HsProb, period: int) -> float:
    """
    计算给定概率分布关于离散参数 shift 的离散 Fisher Information (平均值)。
    参数 theta = shift, 观测值 x = col。
    """
    N = _get_N(prob)
    total_fisher = 0.0
    
    # Since the circuit is shift invariance. we can use shift=0 t0 calculate
    shift = 0 
    next_shift = (shift + 1) % period  
    
    fisher_shift = 0.0
    for col in range(N):
        p_theta = prob(col, shift)
        p_theta_next = prob(col, next_shift)
        
        if p_theta > 1e-12:  
            diff = p_theta_next - p_theta
            fisher_shift += p_theta * (diff / p_theta) ** 2
            
    total_fisher += fisher_shift

    return total_fisher / period


def cross_entropy(prob: Callable[[int, int], float], period: int) -> float:
    """
    计算给定概率分布在相邻离散参数 shift 之间的交叉熵 (平均值)。
    参数 theta = shift, 观测值 x = col。
    """
    N = _get_N(prob)
    total_ce = 0.0
    epsilon = 1e-15  # 防止 log(0)
    
    # Since the circuit is shift invariance. we can use shift=0 t0 calculate
    shift = 0 
    next_shift = (shift + 1) % period
    
    ce_shift = 0.0
    for col in range(N):
        p_theta = prob(col, shift)
        p_theta_next = prob(col, next_shift)
        
        if p_theta > 1e-12:
            ce_shift -= p_theta * np.log(p_theta_next + epsilon)
            
    total_ce += ce_shift

    return total_ce / period