import numpy as np
from typing import Callable
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import altqft.circuits.ph as ph

from altqft.nn.process_qc import make_prob
import numpy as np
from qiskit import QuantumCircuit

def is_shift_invariant(circuit: QuantumCircuit, col: int, period: int) -> bool:
    """
    检查给定的量子电路的矩阵在指定列上是否具有shift invariant。
    """
    prob = make_prob(circuit, period)
    p = prob(col, 0)
    for shift in range(1, period):
        if not np.isclose(prob(col, shift), p):
            return False
    return True

def test_ph_is_sinv():
    """
    随机生成多个 ph 线路，并检查其在 period = 2^n (n < nqubit) 时是否为 True。
    """
    for nqubit in [3, 4, 5]:
        for test_idx in range(3):
            max_layer = np.random.randint(2, nqubit + 1)
            hlayout = np.random.randint(0, max_layer, size=nqubit).tolist()
            phases = np.random.uniform(0, 2 * np.pi, size=nqubit**2)
            
            qc = ph.ph_qc(hlayout, phases)
            

            for n in range(1, nqubit):
                period = 2 ** n
                col = np.random.randint(0, 2**nqubit)
                is_inv = is_shift_invariant(qc, col=col, period=period)
                
                print(f"nqubit={nqubit}, hlayout={hlayout}, period={period} (2^{n}), col={col} "
                      f"-> is_shift_invariant: {is_inv}")
                


if __name__ == "__main__":
    test_ph_is_sinv()