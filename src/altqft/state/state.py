import math
import numpy as np
from qiskit import QuantumCircuit

def initial_state_from_solutions(solutions, n):
    """
    从解的列表生成归一化的量子态矢量
    
    参数:
        solutions: 满足条件的 x 的列表
        n: 量子比特数量 (用于确定态矢量长度)
    
    返回:
        state: 长度为 2**n 的列表，表示量子态的振幅向量
               state[x] = 1/sqrt(k) 如果 x 在 solutions 中
               state[x] = 0 其他情况
    """
    total = 1 << n  # 2**n
    k = len(solutions)
    if k == 0:
        raise ValueError("No solutions provided for state creation")
    
    state = [0.0] * total
    factor = 1.0 / math.sqrt(k)
    
    for x in solutions:
        if x < total:  # 确保 x 在范围内
            state[x] = factor
    
    return state


def qiskit_initial_state(solutions, n):
    """
    将初始态矢量转换为 Qiskit 量子电路的初始态
    
    参数:
        solutions: 满足条件的 x 的列表 (如 [0, 2, 4])
        n: 量子比特数量 (用于确定态矢量长度)
    
    返回:
        qc: Qiskit QuantumCircuit，其初始态为 sum_{x in solutions} |x>
    """

    total = 1 << n  # 2**n
    k = len(solutions)
    if k == 0:
        raise ValueError("No solutions provided for state creation")
    
    state_vector = np.zeros(total, dtype=complex)
    factor = 1.0 / np.sqrt(k)
    
    for x in solutions:
        if x < total:
            state_vector[x] = factor
    
    qc = QuantumCircuit(n)
    
    qc.initialize(state_vector, range(n)) #type: ignore
    
    return qc