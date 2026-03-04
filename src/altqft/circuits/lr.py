import pennylane as qml
import torch
from typing import Callable
import time


def lr_circuit(theta: torch.Tensor, nqubit: int) -> None:
    param_idx = 0
    # 第一层：对偶数索引对比特施加 Hadamard 门
    for i in range(0, nqubit, 2):
        qml.Hadamard(wires=i)

    # 第二层：参数化受控相位层 (偶数控制奇数)
    for i in range(0, nqubit, 2):
        for j in range(1, nqubit, 2):
            # 结合了 QFT 结构的相位项与可训练参数 theta
            phase = torch.pi / 2 ** abs(j - i) + theta[param_idx]
            qml.CPhase(phase, wires=[i, j])
            param_idx += 1

    # 第三层：对奇数索引对比特施加 Hadamard 门
    for i in range(1, nqubit, 2):
        qml.Hadamard(wires=i)


def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算两个概率分布之间的交叉熵"""
    eps = 1e-12
    return -torch.sum(p * torch.log(q + eps))


def make_cost(nqubit: int):

    def bound_circuit(p):
        lr_circuit(p, nqubit)

    # 使用 PennyLane 提供的工具获取电路的幺正矩阵 U(theta)
    matrix_fn = qml.matrix(bound_circuit, wire_order=range(nqubit))  # type: ignore

    def cost_function(params):
        U = matrix_fn(params)  # type: ignore

        # 定义搜索候选周期的范围
        # 对应原逻辑: range(nqubit, max(int(2**(nqubit/4)), nqubit**2))
        start_p = nqubit
        end_p = max(int(2 ** (nqubit / 4)), nqubit**2)

        # 补全逻辑：生成基于当前 U 的不同周期的概率分布池
        # 虽然原片段写的是 map(make_cost, ...)，但在 cost 内部
        # 只有将 U 传递给 make_prob 才能使 loss 对 params 有梯度。
        func_pools = []
        for p in range(start_p, end_p):
            prob_fn = make_prob(U, p)
            # 遍历所有可能的基态 x (0 到 2^n - 1) 获取分布
            # 使用 torch.stack 保持梯度追踪
            dist = torch.stack([prob_fn(x) for x in range(2**nqubit)])
            # 归一化以确保其为合法的概率分布
            # print(dist)
            func_pools.append(dist)

        # 计算相邻周期分布之间的交叉熵并取最大值
        # 对应原逻辑: max( cross_entropy(func_pools[i], func_pools[i+1]) for all i )
        ce_values = [
            cross_entropy(func_pools[i], func_pools[i + 1])
            for i in range(len(func_pools) - 1)
        ]
        # print(ce_values)
        return -torch.logsumexp(-torch.stack(ce_values), dim=0)

    return cost_function, bound_circuit


def make_prob(U: torch.Tensor, period: int) -> Callable:
    def prob(x: int) -> torch.Tensor:
        # 获取矩阵 U 第 x 列中具有特定步长(period)的元素
        # 这在量子周期寻找算法（如 Shor）中对应于特定投影
        period_elements = U[0::period, x]
        # 计算该子空间的概率振幅平方
        return torch.abs(torch.sum(period_elements)) ** 2 / len(period_elements)

    return prob


def train(nqubit: int, epochs: int, lr: float, file_handle, init_params: torch.Tensor = None):
    # 计算当前量子比特数量所需的参数量
    n_params = len(range(0, nqubit, 2)) * len(range(1, nqubit, 2))
    
    # 【需求 1】: 支持接收特殊的初始参数
    if init_params is not None:
        # 使用传入的参数，clone() 是为了避免影响原变量，requires_grad_() 开启梯度
        params = init_params.clone().detach().requires_grad_(True)
    else:
        # 如果没传，默认设为全 0
        params = torch.zeros(n_params, requires_grad=True)
        
    cost_fn, _ = make_cost(nqubit)
    optimizer = torch.optim.Adam([params], lr=lr)
    
    header = f"\n========== 开始训练 {nqubit} 比特线路 ==========\n"
    header += f"参数数量: {n_params} | 学习率: {lr} | 训练轮次: {epochs}\n"
    print(header.strip())
    file_handle.write(header + "\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = cost_fn(params)
        
        if loss.dim() > 0:
            loss = loss.sum()
            
        loss.backward()
        optimizer.step()
        
        # 每隔 5 个 epoch 记录一次，以及第 1 个 epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 将参数转换为普通的 Python list 写入 txt
            current_params_list = params.detach().numpy().tolist()
            params_str = "[" + ", ".join([f"{p:.4f}" for p in current_params_list]) + "]"
            
            # txt 文件中保留参数细节
            log_line = f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f} | Params: {params_str}\n"
            file_handle.write(log_line)
            # 【需求 2】: 控制台不再打印参数，只打印 Loss，保持清爽
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f}")
            
    elapsed = time.time() - start_time
    footer = f"--- 训练阶段完成，耗时: {elapsed:.2f} 秒 ---\n\n"
    print(footer.strip())
    file_handle.write(footer)
    file_handle.flush()
    
    # 【需求 2】: 训练完毕返回参数（供保存或传给下一阶段）
    return params

if __name__ == "__main__":
    FILE_NAME = "training_10_qubits.txt"
    SAVE_MODEL_NAME = "trained_params_10q.pt"
    
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        f.write("=== 量子线路训练数据记录 (两阶段训练) ===\n")
        
        # 【需求 3】: 设置比特数为 10
        n_qubits = 10
        
        # 第一阶段：用 lr = 0.3 训练 100 步
        print(">>> [阶段一] 开始高学习率快速探索 (lr=0.3, epochs=100) <<<")
        params_stage1 = train(nqubit=n_qubits, epochs=80, lr=0.2, file_handle=f)
        
        # 第二阶段：将第一阶段训练好的 params_stage1 传进去，用 lr = 0.1 训练 200 步
        print("\n>>> [阶段二] 开始低学习率精细微调 (lr=0.1, epochs=200) <<<")
        params_stage2 = train(nqubit=n_qubits, epochs=100, lr=0.1, file_handle=f, init_params=params_stage1)
        
    # 保存最终训练出来的参数为 PyTorch 格式 (.pt 文件)
    torch.save(params_stage2, SAVE_MODEL_NAME)
    print(f"\n✅ 所有训练结束！")
    print(f"日志数据已成功保存至: {FILE_NAME}")
    print(f"模型参数已成功保存至: {SAVE_MODEL_NAME}")
