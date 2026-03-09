import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from altqft.circuits  import ph # 导入你的原函数模块

# ==========================================
# 1. 数据预处理与生成
# ==========================================

def counts_to_vector(counts, n, shots):
    """
    将 Qiskit 字典格式的 counts 转换为长度为 2^n 的概率向量
    """
    vec = np.zeros(2**n, dtype=np.float32)
    for bitstr, count in counts.items():
        idx = int(bitstr, 2)
        vec[idx] = count / shots
    return vec

def generate_dataset(n_qubits, num_samples, shots, repeat_factor=5):
    """
    生成用于神经网络训练的数据集，包含 a 和 N 且重复多次以增强特征权重
    """
    X_data = []
    y_data = []
    
    print(f"  -> 开始生成数据集 (目标样本数: {num_samples}, Shots: {shots})...")
    samples_collected = 0
    
    while samples_collected < num_samples:
        # 限制随机数的上下界，防止报错
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
                
            # 运行电路
            _, counts, correct_period = ph.run_lr_on_initial_state(a, c, N, n_qubits, shots=shots)
            
            if counts is not None:
                features = counts_to_vector(counts, n_qubits, shots)
                
                # 将 a 和 N 重复 repeat_factor 次，并适当缩放以防数值过大主导网络
                scale = 1.0 / N
                repeated_params = [float(a) * scale, float(N) * scale] * repeat_factor
                extended_features = np.append(features, repeated_params).astype(np.float32)
                
                X_data.append(extended_features)
                y_data.append(correct_period)
                samples_collected += 1
                
                if samples_collected % 20 == 0:
                    print(f"     已收集 {samples_collected}/{num_samples} 个样本...")
                    
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
# 2. 神经网络模型定义
# ==========================================

class PeriodPredictorNet(nn.Module):
    def __init__(self, n_qubits, repeat_factor=5):
        super(PeriodPredictorNet, self).__init__()
        
        # 输入维度: 2^n (概率分布) + 2 * repeat_factor (参数 a 和 N)
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
# 3. 封装训练函数
# ==========================================

def train(n_qubits, num_train_samples=100, batch_size=16):
    """
    针对特定的量子比特数量训练模型并保存结果
    """
    # 【修改1】: Shots 设定为 n^2 * 1024
    shots = (n_qubits ** 2) * 1024
    
    # 【修改2】: Epoch 设定为 30 * n_qubits
    epochs = 30 * n_qubits
    repeat_factor = 5
    
    # 【修改3】: 确保 data 文件夹存在
    os.makedirs('data', exist_ok=True)
    
    # 1. 生成数据
    X_train, y_train = generate_dataset(n_qubits, num_train_samples, shots, repeat_factor=repeat_factor)
    dataset = QuantumPeriodDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型
    model = PeriodPredictorNet(n_qubits, repeat_factor=repeat_factor)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 3. 开始训练
    print(f"  -> 开始训练网络 (Epochs: {epochs}, Batch Size: {batch_size})...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_preds += batch_y.size(0)
            correct_preds += (predicted == batch_y).sum().item()
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct_preds / total_preds
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"     Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
    print(f"  -> {n_qubits} Qubits 模型训练完成！最终准确率: {accuracy:.2f}%")
    
    # 4. 保存模型与数据到 data 文件夹下
    model_save_path = os.path.join('data', f'period_model_{n_qubits}q.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"  -> 模型已保存至: {model_save_path}\n")
    
    return model

# ==========================================
# 4. 主程序：从 2 qubit 遍历到 10 qubit
# ==========================================

if __name__ == "__main__":
    # 为了演示，设置较少的样本数。
    # 实际应用中，你可能需要将 num_train_samples 设置为 500 或 1000 以上
    samples_per_n = 50  
    
    for n in range(2, 11):
        print("="*50)
        print(f"========== 开始处理 {n} Qubits 的情况 ==========")
        print("="*50)
        
        # 训练并保存模型
        trained_model = train(
            n_qubits=n, 
            num_train_samples=samples_per_n, 
            batch_size=16
        )