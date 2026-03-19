import numpy as np
import torch
from qiskit import QuantumCircuit
from typing import Union
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
    
    # 初始化所有比特都在 rest_qubits 中
    rest_qubits = set(range(nqubit))
    
    for i in range(max_layer):
        curr_hlayer = find_indices(hlayout, i)
        
        # 当前层作为控制比特，剩下的比特需要剔除掉当前层
        rest_qubits = rest_qubits - curr_hlayer
        
        # 将剩余的所有比特作为目标比特环境传入
        ctx = QCEnv(nqubit, rest_qubits, curr_hlayer)
        
        # 参数数量变更为：当前层控制比特数 × 所有剩余目标比特数
        num_para = len(curr_hlayer) * len(rest_qubits)
        
        qc.compose(hp_layer(ctx, phase[idx: idx + num_para]), inplace=True)
        idx += num_para
        
    last_hlayer = find_indices(hlayout, max_layer)
    for q in sorted(last_hlayer):
        qc.h(q)
        
    return qc

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