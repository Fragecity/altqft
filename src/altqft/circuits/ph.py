import numpy as np
from qiskit import QuantumCircuit
from typing import Union
import torch
from dataclasses import dataclass

ArrayInput = Union[list, np.ndarray, torch.Tensor]

@dataclass
class QCEnv:
    nqubit: int
    rest_qubits: set
    curr_hlayer: set

def find_indices(a: list, y: int) -> set:
    return set([i for i, val in enumerate(a) if val == y])

def hp_layer(ctx: QCEnv, parameters: ArrayInput) -> QuantumCircuit:
    qc = QuantumCircuit(ctx.nqubit)
    para_idx = 0
    for control in ctx.curr_hlayer:
        qc.h(control)
        sorted_targets = sorted(list(ctx.rest_qubits - ctx.curr_hlayer))
        for target in sorted_targets:
            qc.cp(parameters[para_idx], control, target)
            para_idx += 1 
    return qc

def ph_qc(hlayout: list, phase: ArrayInput) -> QuantumCircuit:
    nqubit = len(hlayout)
    rest = set(range(nqubit))
    ctx = QCEnv(nqubit, rest, set())
    qc = QuantumCircuit(nqubit)
    idx = 0
    
    for i in range(max(hlayout) + 1):
        hlayer = find_indices(hlayout, i)
        ctx.curr_hlayer = hlayer
        ctx.rest_qubits = rest  
        
        rest = rest - hlayer
        num_para = len(hlayer) * len(rest)
        
        qc.compose(hp_layer(ctx, phase[idx: idx+ num_para]), inplace=True)
        idx += num_para

    return qc

def qft(nqubit: int) -> QuantumCircuit:
    """
    生成标准 QFT 线路（未包含最后的 SWAP 门）
    """
    hlayout = list(range(nqubit))
    phase = np.zeros(int(nqubit*(nqubit-1)/2))
    
    idx = 0
    for control in range(nqubit):
        for target in range(control + 1, nqubit):
            phase[idx] = np.pi / (2 ** (target - control))
            idx += 1
            
    return ph_qc(hlayout, phase)

if __name__ == "__main__":
    qc = ph_qc([0,1,0,1], np.zeros(4))
    print(qc.draw())

    qc = qft(4)
    print(qc.draw())