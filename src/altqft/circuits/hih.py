from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import math
import numpy as np

def lr_circuit_qiskit(nqubit: int) -> QuantumCircuit:
    """
    生成非变分版的 LR 电路（所有 theta = 0）
    参数:
        nqubit: 量子比特数量
    返回:
        Qiskit 量子电路
    """
    # 创建量子电路
    qc = QuantumCircuit(nqubit)
    
    # 第一层：对偶数索引比特施加 Hadamard 门 (0, 2, 4, ...)
    for i in range(0, nqubit, 2):
        qc.h(i)
    
    # 第二层：参数化受控相位层（偶数控制奇数），相位 = π / 2^|j-i|
    for i in range(0, nqubit, 2):  # 偶数索引 (控制比特)
        for j in range(1, nqubit, 2):  # 奇数索引 (目标比特)
            # 计算相位: π / (2^|j-i|)
            phase = math.pi / (2 ** abs(j - i))
            qc.cp(phase, i, j)  # 控制比特 i, 目标比特 j
    
    # 第三层：对奇数索引比特施加 Hadamard 门 (1, 3, 5, ...)
    for i in range(1, nqubit, 2):
        qc.h(i)
    
    return qc

def find_solutions(a, c, N, n):
    """
    找出所有满足 a^x ≡ c (mod N) 的 x (0 ≤ x < 2^n)
    
    参数:
        a: 底数
        c: 目标值 (0 ≤ c < N)
        N: 模数 (N ≥ 1)
        n: 量子比特数量 (x 的二进制表示长度)
    
    返回:
        solutions: 满足条件的 x 的列表 (未归一化)
    """
    # 处理 N=1 的特殊情况
    if N == 1:
        if c != 0:
            raise ValueError("c must be 0 when N=1")
        return list(range(1 << n))  # 所有 x 都满足条件
    
    total = 1 << n  # 2**n
    solutions = []
    
    for x in range(total):
        # 处理 a=0 和 x=0 的边界情况 (0^0 = 1)
        if x == 0:
            val = 1
        else:
            if a == 0:
                val = 0
            else:
                val = pow(a, x, N)
        
        # 检查条件
        if val == c:
            solutions.append(x)
    
    if not solutions:
        raise ValueError(f"No solution found for a={a}, c={c}, N={N}")
    
    return solutions

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
    # 1. 生成归一化的态矢量 (实数)
    total = 1 << n  # 2**n
    k = len(solutions)
    if k == 0:
        raise ValueError("No solutions provided for state creation")
    
    state_vector = np.zeros(total, dtype=complex)
    factor = 1.0 / np.sqrt(k)
    
    for x in solutions:
        if x < total:
            state_vector[x] = factor
    
    # 2. 创建 Qiskit 量子电路
    qc = QuantumCircuit(n)
    
    # 3. 使用 initialize 方法设置初始态
    qc.initialize(state_vector, range(n)) #type: ignore
    
    return qc

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# (这里假设 find_solutions, qiskit_initial_state, lr_circuit_qiskit 
# 已经按照你 hih.py 文件中的定义载入)

def run_lr_on_initial_state(a, c, N, n, shots=1024):
    """
    执行完整流程：查找解 -> 准备初始态 -> 施加 LR 电路 -> 测量
    
    参数:
        a: 底数
        c: 目标值
        N: 模数
        n: 量子比特数量
        shots: 采样/测量次数 (默认: 1024)
    """
    # 1. 寻找满足条件的 x
    try:
        solutions = find_solutions(a, c, N, n)
        print(f"找到的解集 solutions: {solutions}")
    except ValueError as e:
        print(f"错误: {e}")
        return None, None
    
    correct_period = solutions[1] - solutions[0] 

    # 2. 生成初始态电路
    qc_init = qiskit_initial_state(solutions, n)

    # 3. 生成 LR 电路
    lr_circ = lr_circuit_qiskit(n)

    # 4. 组合电路：在初始态之后加上 LR 电路
    complete_circuit = qc_init.compose(lr_circ)

    # 5. 添加测量层 (自动生成名为 'meas' 的经典寄存器)
    complete_circuit.measure_all()

    # 6. 初始化 StatevectorSampler (Qiskit V2 原语)
    sampler = StatevectorSampler()
    
    # 7. 运行电路，传入自定义的 shots 参数
    job = sampler.run([complete_circuit], shots=shots)
    result = job.result()
    
    # 8. 获取测量计数值
    counts = result[0].data.meas.get_counts()

    return complete_circuit, counts, correct_period



