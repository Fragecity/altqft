import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from altqft.circuits import ph # 导入你的量子电路函数

# ==========================================
# 1. 依赖函数与网络结构 (需与训练代码保持一致)
# ==========================================

def counts_to_vector(counts, n, shots):
    vec = np.zeros(2**n, dtype=np.float32)
    for bitstr, count in counts.items():
        idx = int(bitstr, 2)
        vec[idx] = count / shots
    return vec

def generate_test_dataset(n_qubits, num_samples, shots, repeat_factor=5):
    """
    专门用于生成测试集：逻辑与训练集生成相同，但由于调用时机不同，
    np.random 会生成与训练时不同的 (a, c, N) 组合（未见数据）。
    """
    X_data = []
    y_data = []
    
    samples_collected = 0
    while samples_collected < num_samples:
        N = max(int(2 ** (n_qubits / 4)), n_qubits**2)
        lower_bound = int(N/4 + 1)
        upper_bound = N - 3
        
        if lower_bound >= upper_bound:
            if N <= 3:
                continue
            a = np.random.randint(2, N)
        else:
            a = np.random.randint(lower_bound, upper_bound)
        
        try:
            solutions_1 = ph.find_solutions(a, 1, N, n_qubits)
            if len(solutions_1) < 2:
                continue
            period = solutions_1[1] - solutions_1[0]
            if period <= 1:
                continue
            c = np.random.randint(1, period)
            
            sols = ph.find_solutions(a, c, N, n_qubits)
            if len(sols) < 2:
                continue
                
            _, counts, correct_period = ph.run_lr_on_initial_state(a, c, N, n_qubits, shots=shots)
            
            if counts is not None:
                features = counts_to_vector(counts, n_qubits, shots)
                scale = 1.0 / N
                repeated_params = [float(a) * scale, float(N) * scale] * repeat_factor
                extended_features = np.append(features, repeated_params).astype(np.float32)
                
                X_data.append(extended_features)
                y_data.append(correct_period)
                samples_collected += 1
                
        except ValueError:
            continue

    return np.array(X_data), np.array(y_data)

class QuantumPeriodDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PeriodPredictorNet(nn.Module):
    def __init__(self, n_qubits, repeat_factor=5):
        super(PeriodPredictorNet, self).__init__()
        input_size = (2 ** n_qubits) + 2 * repeat_factor
        max_possible_period = 2 ** n_qubits  
        
        hidden_1 = max(128, input_size)
        hidden_2 = max(64, input_size // 2)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, max_possible_period) 
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. 测试主流程与绘图
# ==========================================

def test_models_and_plot(max_qubits=10, num_test_samples=50):
    """
    遍历测试已保存的模型，记录准确率并绘图。
    """
    qubits_list = []
    accuracy_list = []
    
    print("\n========== 开始在全新测试集上评估模型 ==========")
    
    for n in range(2, max_qubits + 1):
        model_path = os.path.join('data', f'period_model_{n}q.pth')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"  -> 未找到 {n} Qubits 的模型文件 ({model_path})，跳过...")
            continue
            
        print(f"\n正在测试 {n} Qubits 模型...")
        
        # 1. 准备参数
        shots = (n ** 2) * 1024
        repeat_factor = 5
        
        # 2. 加载模型结构和权重
        model = PeriodPredictorNet(n, repeat_factor=repeat_factor)
        model.load_state_dict(torch.load(model_path))
        
        # 【关键】将模型设置为评估模式 (关闭 Dropout/BatchNorm 的训练行为)
        model.eval()
        
        # 3. 生成未见过的测试数据
        print(f"  -> 生成 {num_test_samples} 个全新的测试样本 (Shots={shots})...")
        X_test, y_test = generate_test_dataset(n, num_test_samples, shots, repeat_factor=repeat_factor)
        test_dataset = QuantumPeriodDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 4. 在测试集上评估 (关闭梯度计算以节省内存并加速)
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_preds += batch_y.size(0)
                correct_preds += (predicted == batch_y).sum().item()
                
        # 5. 计算并记录准确率
        accuracy = 100 * correct_preds / total_preds
        print(f"  -> {n} Qubits 测试集准确率: {accuracy:.2f}%")
        
        qubits_list.append(n)
        accuracy_list.append(accuracy)

    # ==========================================
    # 3. 绘制准确率折线图
    # ==========================================
    if not qubits_list:
        print("没有测试任何模型，无法绘图。")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(qubits_list, accuracy_list, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.title('Neural Network Test Accuracy vs. Number of Qubits', fontsize=14)
    plt.xlabel('Number of Qubits (n)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(qubits_list) # 确保 x 轴只显示整数刻度
    plt.ylim(0, 105) # y 轴范围 0-100%
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    os.makedirs('data', exist_ok=True)
    plot_path = os.path.join('data', 'test_accuracy_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n========== 测试完成 ==========")
    print(f"  -> 准确率走势图已保存至: {plot_path}")
    
    # 如果是在图形界面/Jupyter下运行，可显示图表
    # plt.show() 

if __name__ == "__main__":
    # 为了测试速度，样本数暂设为 50。若要获得更稳定的准确率评估，请增加至 100 或 200 以上。
    test_samples_per_n = 50 
    test_models_and_plot(max_qubits=10, num_test_samples=test_samples_per_n)