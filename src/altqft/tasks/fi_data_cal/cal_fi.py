from dataclasses import dataclass
from typing import List, Iterator, Optional
from qiskit import QuantumCircuit
import random
import pickle
import os

from altqft.nn.process_qc import min_fi
from altqft.circuits.ph import qft
from altqft.circuits.ph import ph_phase

@dataclass
class FiMetaData:
    circuit_type: str
    nqubit_range: List[int]
    repeat: int
    hlayout: Optional[List[int]] = None

@dataclass
class FiData:
    circuit_type: str
    nqubit: int
    fi_val: float
    hlayout: Optional[List[int]] = None


def random_hlayout(nqubit: int) -> list[int]:
    hlayout = [0]
    current_max = 0

    for _ in range(1, nqubit):
        x = random.randint(0, current_max + 1)
        hlayout.append(x)
        current_max = max(current_max, x)

    return hlayout

def circuit_router(circuit_type: str, nqubit: int, hlayout: Optional[List[int]] = None) -> Iterator[QuantumCircuit]:
    """
    根据给定的参数生成对应的量子电路。
    使用 yield 无限生成，配合外层 repeat 循环调用 next()。
    """
    while True:
        if circuit_type.lower() == "qft":
            yield qft(nqubit)
        elif circuit_type.lower() == "ph":
            if hlayout is None:
                raise ValueError("ph 电路需要提供 hlayout")
            yield ph_phase(hlayout)
        # elif circuit_type.lower() == "random":
        #     yield random_circuit(nqubit, nlayer)
     
        else:
            raise ValueError(f"暂不支持的电路类型: {circuit_type}")
        

def get_fi(meta_data: FiMetaData) -> List[FiData]:
    """
    遍历元数据中指定的比特数和层数，生成电路并计算最小 Fisher Information。
    """
    res: List[FiData] = []
    
    for nqubit in meta_data.nqubit_range:
        qc_itr = circuit_router(meta_data.circuit_type, nqubit, meta_data.hlayout)
            
        upper_bound = min(max(int(2 ** (nqubit / 4)), nqubit ** 2), int(nqubit**2/2))
        p_range = range(nqubit, upper_bound + 1)
            
        for _ in range(meta_data.repeat):
            qc = next(qc_itr)
        
            val = min_fi(qc, period_range=p_range)
                
            fidata = FiData(
                    circuit_type=meta_data.circuit_type,
                    nqubit=nqubit,
                    fi_val=val,
                    hlayout=meta_data.hlayout
                )
            res.append(fidata)
                
    return res

if __name__ == "__main__":


    all_results = []

    # 简单的本地测试用例
    meta = FiMetaData(
        circuit_type="qft",
        nqubit_range=list(range(3,6)),
        repeat=1
    )
    results = get_fi(meta)
    all_results.extend(results)
    for r in results:
        print(r)


    # use ph to simulate qft
    for nqubit in range(3, 6):
        meta = FiMetaData(
            circuit_type="ph",
            nqubit_range=[nqubit],
            hlayout=list(range(nqubit)),
            repeat=1
    )

        results = get_fi(meta)
        all_results.extend(results)

        for r in results:
            print(r)
    


    # random hlayout
    for nqubit in range(3, 6):
        hlayout = random_hlayout(nqubit)

        meta = FiMetaData(
            circuit_type="ph",
            nqubit_range=[nqubit],
            hlayout=hlayout,
            repeat=1
        )

        results = get_fi(meta)
        all_results.extend(results)

        for r in results:
            print(r)    

    save_dir = "data/shared"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join("data/shared", "fi_results.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Saved to {save_path}")