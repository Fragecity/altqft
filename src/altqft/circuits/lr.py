import pennylane as qml
import torch
from typing import Callable


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
            dist = dist / (torch.sum(dist) + 1e-12)
            func_pools.append(dist)

        # 计算相邻周期分布之间的交叉熵并取最大值
        # 对应原逻辑: max( cross_entropy(func_pools[i], func_pools[i+1]) for all i )
        ce_values = [
            cross_entropy(func_pools[i], func_pools[i + 1])
            for i in range(len(func_pools) - 1)
        ]

        return torch.logsumexp(-torch.stack(ce_values), dim=0)

    return cost_function, bound_circuit


def make_prob(U: torch.Tensor, period: int) -> Callable:
    def prob(x: int) -> torch.Tensor:
        # 获取矩阵 U 第 x 列中具有特定步长(period)的元素
        # 这在量子周期寻找算法（如 Shor）中对应于特定投影
        period_elements = U[0::period, x]
        # 计算该子空间的概率振幅平方
        return torch.abs(torch.sum(period_elements)) ** 2 / len(period_elements)

    return prob


def train(nqubit: int, epochs: int, lr: float, file_handle):
    """
    通用训练函数，并将结果写入传入的 file_handle
    """
    n_params = len(range(0, nqubit, 2)) * len(range(1, nqubit, 2))

    # 初始参数全设为 0
    params = torch.zeros(n_params, requires_grad=True)
    cost_fn, _ = make_cost(nqubit)
    optimizer = torch.optim.Adam([params], lr=lr)

    header = f"\n========== 开始训练 {nqubit} 比特线路 ==========\n"
    header += f"参数数量: {n_params} | 学习率: {lr} | 训练轮次: {epochs}\n"
    print(header.strip())
    file_handle.write(header)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = cost_fn(params)

        if loss.dim() > 0:
            loss = loss.sum()

        loss.backward()
        optimizer.step()

        # 将参数转换为普通的 Python list 以便写入 txt
        current_params_list = params.detach().numpy().tolist()
        params_str = "[" + ", ".join([f"{p:.4f}" for p in current_params_list]) + "]"

        # 记录每一步的数据
        log_line = (
            f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f} | Params: {params_str}\n"
        )
        file_handle.write(log_line)

        # 控制台仅每 10 轮打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(log_line.strip())

    footer = f"--- {nqubit} 比特训练完成 ---\n\n"
    print(footer.strip())
    file_handle.write(footer)
    file_handle.flush()  # 强制写入硬盘


# --- 测试代码 ---
if __name__ == "__main__":
    # 配置训练参数
    EPOCHS = 50  # 训练轮次
    LR = 0.1  # 学习率
    FILE_NAME = "training_landscape.txt"

    # 打开文件，使用 'w' 模式会覆盖旧文件。想追加可以改为 'a'
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        f.write("=== 量子线路训练数据记录 (Loss & Landscape) ===\n")

        # 从 2 比特遍历到 10 比特
        # 注意：为了避免长时间卡住，如果你发现 8 或 10 比特太慢，可以随时使用 Ctrl+C 中断
        for qubits in range(2, 11):
            train(nqubit=qubits, epochs=EPOCHS, lr=LR, file_handle=f)

    print(f"所有训练结束！数据已成功保存至 {FILE_NAME}")
