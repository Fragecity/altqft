import pennylane as qml
import torch
from typing import Callable
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# 1. 核心线路与代价函数定义
# ==========================================

def lr_circuit(theta: torch.Tensor, nqubit: int) -> None:
    param_idx = 0
    # 第一层
    for i in range(0, nqubit, 2):
        qml.Hadamard(wires=i)
    
    # 第二层
    for i in range(0, nqubit, 2):
        for j in range(1, nqubit, 2):
            phase = torch.pi / 2**abs(j - i) + theta[param_idx]
            qml.CPhase(phase, wires=[i, j])
            param_idx += 1
            
    # 第三层
    for i in range(1, nqubit, 2):
        qml.Hadamard(wires=i)

def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -torch.sum(p * torch.log(q + eps))

def make_prob(U: torch.Tensor, period: int) -> Callable:
    def prob(x: int) -> torch.Tensor:
        period_elements = U[0::period, x]
        return torch.abs(torch.sum(period_elements))**2 / len(period_elements)
    return prob

def make_cost(nqubit: int):
    def bound_circuit(p):
        lr_circuit(p, nqubit)
    
    matrix_fn = qml.matrix(bound_circuit, wire_order=range(nqubit)) 

    def cost_function(params):
        U = matrix_fn(params)
        
        start_p = nqubit
        end_p = max(int(2**(nqubit/4)), nqubit**2)
        
        func_pools = []
        for p in range(start_p, end_p):
            prob_fn = make_prob(U, p)
            dist = torch.stack([prob_fn(x) for x in range(2**nqubit)])
            dist = dist / (torch.sum(dist) + 1e-12)
            func_pools.append(dist)
            
        if len(func_pools) < 2:
            return torch.tensor(0.0, requires_grad=False)
        
        ce_values = [
            cross_entropy(func_pools[i], func_pools[i+1]) 
            for i in range(len(func_pools) - 1)
        ]
        
        # 平滑的 Max-Min 逻辑
        return min(torch.stack(ce_values))
        
    return cost_function

# ==========================================
# 2. 评估函数与绘图逻辑
# ==========================================

def evaluate_and_plot(min_q: int = 2, max_q: int = 14, step: int = 2, save_name: str = "init_cost_landscape.png"):
    qubit_list = list(range(min_q, max_q + 1, step))
    loss_list = []
    
    print(f"=== 开始评估初始参数下的 Cost (从 {min_q} 到 {max_q} 比特) ===")
    
    for q in qubit_list:
        start_time = time.time()
        
        # 1. 计算当前比特数需要的参数量
        n_params = len(range(0, q, 2)) * len(range(1, q, 2))
        
        # 2. 仅使用原始参数 (全0)，不需要计算梯度
        params = torch.zeros(n_params, requires_grad=False)
        
        # 3. 构造代价函数并计算 Loss
        cost_fn = make_cost(q)
        
        # 禁用梯度追踪以节省内存并加速前向传播
        with torch.no_grad():
            loss = cost_fn(params)
            # 如果返回值有多维，确保取标量
            if loss.dim() > 0:
                loss = loss.sum()
            loss_val = loss.item()
            
        loss_list.append(loss_val)
        
        elapsed = time.time() - start_time
        print(f"Qubits: {q:2d} | 初始 Loss: {loss_val:.6f} | 耗时: {elapsed:.2f} 秒")
        
    # 4. 绘制横轴为 Qubit、纵轴为 Loss 的图像
    plt.figure(figsize=(9, 6))
    plt.plot(qubit_list, loss_list, marker='s', markersize=8, color='b', linewidth=2, label="Initial Cost")
    
    # 设置图表样式
    plt.title('Initial Cost Landscape vs Number of Qubits', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Qubits', fontsize=12)
    plt.ylabel('Initial Loss (Smoothed Max-Min CE)', fontsize=12)
    plt.xticks(qubit_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ 评估完成！折线图已保存至: {os.path.abspath(save_name)}")

if __name__ == "__main__":

    evaluate_and_plot(min_q=2, max_q=12, step=2, save_name="init_cost_landscape.png")