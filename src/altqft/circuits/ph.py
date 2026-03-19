import numpy as np
from qiskit import QuantumCircuit
from typing import Union
from collections import Counter
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
    sorted_targets = sorted(list(ctx.rest_qubits - ctx.curr_hlayer))
    
    for control in ctx.curr_hlayer:
        qc.h(control)
        for target in sorted_targets:
            qc.cp(parameters[para_idx], control, target)
            para_idx += 1
    return qc

def ph_qc(hlayout: list, phase: ArrayInput) -> QuantumCircuit:
    nqubit = len(hlayout)
    qc = QuantumCircuit(nqubit)
    idx = 0
    max_layer = max(hlayout)
    
    for i in range(max_layer):
        curr_hlayer = find_indices(hlayout, i)
        next_hlayer = find_indices(hlayout, i + 1)
        ctx = QCEnv(nqubit, next_hlayer, curr_hlayer)
        num_para = len(curr_hlayer) * len(next_hlayer)
        qc.compose(hp_layer(ctx, phase[idx: idx + num_para]), inplace=True)
        idx += num_para
        
    last_hlayer = find_indices(hlayout, max_layer)
    for q in sorted(last_hlayer):
        qc.h(q)
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

def ph_phase(hlayout: list) -> QuantumCircuit:
    """
    按照 ph_qc 的连接顺序生成 phase
    """
    nqubit = len(hlayout)
    phases = []
    rest = set(range(nqubit))

    for layer in range(max(hlayout) + 1):
        hlayer = [i for i, x in enumerate(hlayout) if x == layer]
        rest = rest - set(hlayer)
        targets = sorted(rest)
        controls = hlayer
        for control in controls:
            for target in targets:
                phases.append(np.pi / (2 ** abs(target - control)))
    
    phases = np.array(phases)
    return ph_qc(hlayout, phases)

if __name__ == "__main__":
    qc = ph_qc([0,1,0,1], np.zeros(4))
    print(qc.draw())
    qc = ph_phase([0,1,0,1])
    print(qc.draw())
    qc = ph_phase([0,1,2,0,1,2])
    print(qc.draw())
    qc = qft(4)
    print(qc.draw())