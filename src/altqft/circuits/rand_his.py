import math
from random import random
from qiskit import QuantumCircuit

from altqft.checker.shift_inv import is_shift_invariant

def rand_h_skeleton(nqubits: int, nlayers: int) -> list:
    """
    生成Hadamard门的分布。每个比特上仅可能有一个Hadamard门，且每层的Hadamard门分布随机。
     
    Args:
        nqubits (int): 量子比特数
        nlayers (int): 线路层数

    Returns:
        list: 每个qubit的Hadamard门所在层的位置。

    Example:
        >>> rand_h_skeleton(nqubits=3, nlayers=2)
        [0, 1, 0]  # 表示第0和第2个比特的H门在第0层，第1个比特的H门在第1层。
    """
    return [int(random() * nlayers) for _ in range(nqubits)]

def rand_cp_block(nqubits: int) -> QuantumCircuit:
    """
    生成随机的受控相位门区块。
    
    n比特上的随机cp门，随机的相位角度。以50%的概率在任意两两比特间施加CP门。

    Args:
        nqubits (int): 量子比特数

    Returns:
        QuantumCircuit: 包含随机CP门的量子线路。
    """
    qc = QuantumCircuit(nqubits)
    for i in range(nqubits):
        for j in range(i + 1, nqubits):
            if random() > 0.5:
                theta = random() * 2 * math.pi
                qc.cp(theta, i, j)
    return qc

def build_rand_h_circuit(nqubits: int, nlayers: int) -> tuple[QuantumCircuit, list]:
    """
    构建完整的随机Hadamard电路，并返回其Hadamard分布。

    Args:
        nqubits (int): 量子比特数
        nlayers (int): 线路层数

    Returns:
        tuple[QuantumCircuit, list]: (构建好的随机Hadamard电路, Hadamard门所在层的列表)
    """
    qc = QuantumCircuit(nqubits)
    h_skl = rand_h_skeleton(nqubits, nlayers)
    
    for layer in range(nlayers):
        # 如果 h_skl[i] == layer，则在第 i 个比特上添加 Hadamard 门
        for i in range(nqubits):
            if h_skl[i] == layer:
                qc.h(i)
        
        # 将随机 CP Block 拼接到主电路上
        cp_block = rand_cp_block(nqubits)
        qc.compose(cp_block, inplace=True)

    return qc, h_skl  # 修改为返回元组


def build_rand_sinv_circuit(nqubits: int, nlayers: int) -> tuple[QuantumCircuit, list]:
    """
    构建完整的具有 shift invariant 性质的随机 Hadamard 电路，并返回其Hadamard分布。

    Args:
        nqubits (int): 量子比特数
        nlayers (int): 线路层数

    Returns:
        tuple[QuantumCircuit, list]: (满足性质的电路, Hadamard门所在层的列表)
    """
    attempt = 0
    while True:
        attempt += 1
        # 解包 build_rand_h_circuit 返回的两个值
        qc, h_skl = build_rand_h_circuit(nqubits, nlayers)
        
        # 检查该电路是否满足 shift invariant 性质 (以 col=0, period=2 为标准)
        if is_shift_invariant(qc, col=0, period=2):
            print(f"成功！在第 {attempt} 次尝试后生成了满足条件的电路。")
            return qc, h_skl
        
        # 进度打印
        if attempt % 50 == 0:
            print(f"已尝试 {attempt} 次，正在继续生成...")


if __name__ == "__main__":
    # 设置参数
    num_qubits = 4
    num_layers = 4
    # from qiskit import transpile



    # print(f"--- 正在生成完整的随机 sinv 电路 (Layers: {num_layers}) ---")
    # full_circuit, _ = build_rand_sinv_circuit(num_qubits, num_layers)
    # print(transpile(full_circuit, optimization_level=2).draw(output='text'))
    qc,_ = build_rand_h_circuit(num_qubits,num_layers)
    print(qc)

    res = is_shift_invariant(qc,2,4)
    print(res)