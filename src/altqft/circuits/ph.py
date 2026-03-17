import numpy as np
from qiskit import QuantumCircuit
from typing import Union
from collections import Counter
import torch
from dataclasses import dataclass

ArrayInput = Union[list, np.ndarray, torch.Tensor]
# annotation for array input, which can be a list, numpy array, or torch tensor

@dataclass
# decorator for quantum circuit environment, which holds the current state of the circuit construction
class QCEnv:
    nqubit: int
    rest_qubits: set
    curr_hlayer: set

def find_indices(a: list, y: int) -> set:
    # find the indices of elements in list a that are equal to y and return them as a set
    return set([i for i, val in enumerate(a) if val == y])

def hp_layer(ctx: QCEnv, parameters: ArrayInput) -> QuantumCircuit:
    # create quantum circuit for a single layer of Hadamard and controlled phase gates based on the current context and parameters
    qc = QuantumCircuit(ctx.nqubit)
    para_idx = 0
    for control in ctx.curr_hlayer:
        qc.h(control)
        # put hadamard gates on the control qubits in the current layer
        sorted_targets = sorted(list(ctx.rest_qubits - ctx.curr_hlayer))
        # sort qubits that are not in the current but in the rest
        for target in sorted_targets:
            qc.cp(parameters[para_idx], control, target)
            # control phase gates
            para_idx += 1 
    return qc

def ph_qc(hlayout: list, phase: ArrayInput) -> QuantumCircuit:
    # hlayout gives order of controls
    nqubit = len(hlayout)
    rest = set(range(nqubit))
    ctx = QCEnv(nqubit, rest, set())
    qc = QuantumCircuit(nqubit)
    idx = 0
    
    for i in range(max(hlayout) + 1):
        # find the maximum number of layers
        hlayer = find_indices(hlayout, i)

        ctx.curr_hlayer = hlayer
        # the current qubit controlling
        ctx.rest_qubits = rest  
        rest = rest - hlayer
        # the real rest
        num_para = len(hlayer) * len(rest)
        
        qc.compose(hp_layer(ctx, phase[idx: idx+ num_para]), inplace=True)
        # cut the needed phase segment phase[idx: idx+ num_para]
        idx += num_para

    return qc


def ph_qc2(hlayout: list, phase: ArrayInput) -> QuantumCircuit:
    # hlayout gives order of controls
    nqubit = len(hlayout)
    qc = QuantumCircuit(nqubit)
    idx = 0
    max_layer = max(hlayout)

    for i in range(max_layer):
        # 当前层
        curr_hlayer = find_indices(hlayout, i)
        # 下一层
        next_hlayer = find_indices(hlayout, i + 1)

        ctx = QCEnv(nqubit, next_hlayer, curr_hlayer)

        # 相邻层之间的参数数量
        num_para = len(curr_hlayer) * len(next_hlayer)

        qc.compose(hp_layer(ctx, phase[idx: idx + num_para]), inplace=True)
        idx += num_para

    # 最后一层补 Hadamard
    last_hlayer = find_indices(hlayout, max_layer)
    for q in sorted(last_hlayer):
        qc.h(q)

    return qc





def qft(nqubit: int) -> QuantumCircuit:
    """
    生成标准 QFT 线路（未包含最后的 SWAP 门）
    no hadamard gates, only phase 
    """
    hlayout = list(range(nqubit))
    # number of phase gates 
    phase = np.zeros(int(nqubit*(nqubit-1)/2))
    
    idx = 0
    for control in range(nqubit):
        for target in range(control + 1, nqubit):
            phase[idx] = np.pi / (2 ** (target - control))
            idx += 1
            
    return ph_qc(hlayout, phase)


def ph_phase(hlayout: list) -> QuantumCircuit:
        
    """
    生成ph线路所需要的phase
    结合已有的ph_qc函数，生成量子线路
    no hadamard gates, only phase 
    """


    # 分层
    groups = [[i for i, x in enumerate(hlayout) if x == layer] for layer in range(max(hlayout)+1)]

    # 建连接
    connections = [
    (layer, control, target)
    for layer in range(len(groups)-1)
    for control in groups[layer]
    for target in groups[layer+1]]



    # 对每条连接计算 phase = pi / 2^(j-i)


    phases = np.array([np.pi / (2 ** abs(target - control)) 
                       for layer, control, target in connections])
    
    return ph_qc2(hlayout, phases)
    




if __name__ == "__main__":
    qc = ph_qc([0,1,0,1], np.zeros(4))
    print(qc.draw())

    qc = ph_phase([0,1,0,1])
    print(qc.draw())

    qc = ph_phase([0,1,2,0,1,2])
    print(qc.draw())

    # qc = qft(4)
    # print(qc.draw())