import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from altqft.circuits import ph # 导入你的原函数模块

# ==========================================
# 1. 依赖函数与网络结构 (必须与训练时完全一致)
# ==========================================

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

def counts_to_vector(counts, n, shots):
    vec = np.zeros(2**n, dtype=np.float32)
    for bitstr, count in counts.items():
        idx = int(bitstr, 2)
        vec[idx] = count / shots
    return vec

# ==========================================
# 2. 带有“扰动”的测试数据生成器
# ==========================================

def generate_perturbed_test_dataset(n_qubits, num_samples, shots, repeat_factor=5, perturb_type='shuffle', noise_level=0.1):
    """
    生成测试集，并对测量出来的分布进行物理扰动。
    
    perturb_type: 
        - 'shuffle': 完全打乱测量分布（最严格的作弊测试）。
        - 'noise': 加入高斯噪声，模拟硬件误差。
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
                # 获取原始的量子测量概率分布
                features = counts_to_vector(counts, n_qubits, shots)
                
                # ==========================================
                # 核心：对测量特征进行扰动
                # ==========================================
                if perturb_type == 'shuffle':
                    # 完全打乱量子态分布，销毁傅里叶图样信息
                    np.random.shuffle(features)
                    
                elif perturb_type == 'noise':
                    # 加入高斯白噪声并重新归一化
                    noise = np.random.normal(0, noise_level, size=features.shape)
                    features = np.clip(features + noise, 0, 1) # 防止出现负概率
                    if features.sum() > 0:
                        features = features / features.sum()
                
                # a 和 N 作为“潜在的作弊线索”保持不变
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

# ==========================================
# 3. 评估与绘图逻辑
# ==========================================

def evaluate_and_plot_robustness(max_qubits=10, num_test_samples=50, perturb_type='shuffle'):
    qubits_list = []
    accuracy_list = []
    
    print(f"\n========== 开始抗扰动测试 (扰动模式: {perturb_type}) ==========")
    print("目标：测试网络是否偷偷绕过了量子电路的输出，直接通过 a 和 N 计算了周期。")
    
    for n in range(2, max_qubits + 1):
        model_path = os.path.join('data', f'period_model_{n}q.pth')
        
        if not os.path.exists(model_path):
            print(f"  -> 未找到 {n} Qubits 的模型，跳过...")
            continue
            
        print(f"\n评估 {n} Qubits 模型...")
        shots = (n ** 2) * 1024
        repeat_factor = 5
        
        # 加载模型
        model = PeriodPredictorNet(n, repeat_factor=repeat_factor)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 生成包含扰动的数据
        print(f"  -> 生成 {num_test_samples} 个携带 '{perturb_type}' 扰动的样本...")
        X_test, y_test = generate_perturbed_test_dataset(
            n, num_test_samples, shots, repeat_factor=repeat_factor, perturb_type=perturb_type
        )
        
        test_dataset = QuantumPeriodDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 评估准确率
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_preds += batch_y.size(0)
                correct_preds += (predicted == batch_y).sum().item()
                
        accuracy = 100 * correct_preds / total_preds
        print(f"  -> {n} Qubits 受扰动后的测试准确率: {accuracy:.2f}%")
        
        qubits_list.append(n)
        accuracy_list.append(accuracy)

    # 绘图
    if not qubits_list:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(qubits_list, accuracy_list, marker='s', linestyle='-', color='r', linewidth=2, markersize=8, label=f'Perturbed ({perturb_type})')
    
    plt.title(f'NN Test Accuracy with Perturbed Quantum Data ({perturb_type})', fontsize=14)
    plt.xlabel('Number of Qubits (n)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(qubits_list)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    os.makedirs('data', exist_ok=True)
    plot_path = os.path.join('data', f'robustness_plot_{perturb_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n========== 测试完成 ==========")
    print(f"  -> 分析图表已保存至: {plot_path}")

if __name__ == "__main__":
    # 使用 'shuffle' 彻底打乱量子测量结果，看网络是否还在依靠 a 和 N 蒙对答案。
    evaluate_and_plot_robustness(max_qubits=10, num_test_samples=50, perturb_type='shuffle')