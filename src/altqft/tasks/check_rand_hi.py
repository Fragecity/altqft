from qiskit import QuantumCircuit
from altqft.circuits.rand_his import build_rand_h_circuit
from altqft.checker.shift_inv import is_shift_invariant


# 这段代码说明了不是所有的随机Hadamard电路都具有shift invariant。通过构建一个随机Hadamard电路，并使用is_shift_invariant函数检查其是否具有shift invariant，我们可以得出结论。
def is_rand_hi_sinv(nqubits: int, nlayers: int) -> bool:
    """
    检查随机Hadamard电路是否具有shift invariant。
    
    Args:
        nqubits (int): 量子比特数
        nlayers (int): 线路层数

    Returns:
        bool: 如果随机Hadamard电路具有shift invariant，则返回True；否则返回False。
    
    Example:
        >>> is_rand_hi_sinv(nqubits=3, nlayers=2)
        True  # 如果构建的随机Hadamard电路具有shift invariant
    """
    qc, _ = build_rand_h_circuit(nqubits, nlayers)
    return is_shift_invariant(qc, col=0, period=2)




# 现在测试，如果具有shift invariant，那么 h_skl 是不是一个
if __name__ == "__main__":
    nqubits = 4
    nlayers = 4
    result = is_rand_hi_sinv(nqubits, nlayers)
    print(f"随机Hadamard电路是否具有shift invariant: {result}")

