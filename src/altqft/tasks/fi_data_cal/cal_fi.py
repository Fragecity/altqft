from dataclasses import dataclass
from typing import List, Iterator
from qiskit import QuantumCircuit

from altqft.nn.process_qc import min_fi
from altqft.circuits.ph import qft

@dataclass
class FiMetaData:
    circuit_type: str
    nqubit_range: List[int]
    nlayer_range: List[int]
    repeat: int

@dataclass
class FiData:
    circuit_type: str
    nqubit: int
    nlayer: int
    fi_val: float

def circuit_router(circuit_type: str, nqubit: int, nlayer: int) -> Iterator[QuantumCircuit]:
    """
    根据给定的参数生成对应的量子电路。
    使用 yield 无限生成，配合外层 repeat 循环调用 next()。
    """
    while True:
        if circuit_type.lower() == "qft":
            yield qft(nqubit)
        else:
            raise ValueError(f"暂不支持的电路类型: {circuit_type}")

def get_fi(meta_data: FiMetaData) -> List[FiData]:
    """
    遍历元数据中指定的比特数和层数，生成电路并计算最小 Fisher Information。
    """
    res: List[FiData] = []
    
    for nqubit in meta_data.nqubit_range:
        for nlayer in meta_data.nlayer_range:
            qc_itr = circuit_router(meta_data.circuit_type, nqubit, nlayer)
            
            upper_bound = min(max(int(2 ** (nqubit / 4)), nqubit ** 2), int(nqubit**2/2))
            p_range = range(nqubit, upper_bound + 1)
            
            for _ in range(meta_data.repeat):
                qc = next(qc_itr)
        
                val = min_fi(qc, period_range=p_range)
                
                fidata = FiData(
                    circuit_type=meta_data.circuit_type,
                    nqubit=nqubit,
                    nlayer=nlayer,
                    fi_val=val
                )
                res.append(fidata)
                
    return res

if __name__ == "__main__":
    # 简单的本地测试用例
    meta = FiMetaData(
        circuit_type="qft",
        nqubit_range=list(range(3,11)),
        nlayer_range=[-1], 
        repeat=1
    )
    results = get_fi(meta)
    for r in results:
        print(r)