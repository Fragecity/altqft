import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from altqft.circuits import ph # 导入你的量子电路模块

# ==========================================
# 1. 设置想要测试的参数
# ==========================================
n_qubits = 10
a = 97
c = 1
N = 111
shots = 1000000

# ==========================================
# 2. 运行电路并获取 counts
# ==========================================
n_qubits = 10
a = 97
c = 1
N = 111
shots = 1000000

print(f"正在运行 {n_qubits} Qubits 的量子电路模拟 (shots={shots})...")
print("这可能需要几十秒的时间，请稍候...")

# ==========================================
# 2. 运行电路并获取 counts
# ==========================================
circuit, counts, correct_period = ph.run_lr_on_initial_state(a, c, N, n_qubits, shots=shots)

# if counts is not None:
#     print(f"真实周期: {correct_period}")
    
#     # ==========================================
#     # 3. 数据处理: 去除 0000 且转化为十进制
#     # ==========================================
#     # 生成对应长度的全 0 字符串
#     zero_state = '0' * n_qubits 
    
#     # 忽略 0000 这个比特态
#     if zero_state in counts:
#         del counts[zero_state]
        
#     # 将字典的键从 二进制字符串 转换为 十进制整数
#     decimal_counts = {int(k, 2): v for k, v in counts.items()}
    
#     # 按十进制数值大小对字典进行排序，保证画图时 X 轴从左到右递增
#     sorted_counts = dict(sorted(decimal_counts.items()))
    
#     # ==========================================
#     # 4. 开始绘图
#     # ==========================================
#     plt.figure(figsize=(16, 6))
    
#     x_vals = list(sorted_counts.keys())
#     y_vals = list(sorted_counts.values())
    
#     # 绘制柱状图 (由于有 1024 个状态，把柱宽 width 调宽一些可以看清峰值)
#     plt.bar(x_vals, y_vals, color='royalblue', width=2.0)
    
#     plt.title(f'Quantum Circuit Measurement Counts vs Decimal State\n(a={a}, c={c}, N={N}, n={n_qubits}) | Zero State Excluded', fontsize=14)
#     plt.xlabel('Measured State (Decimal)', fontsize=12)
#     plt.ylabel(f'Counts (out of {shots} shots)', fontsize=12)
    
#     # 设置 X 轴的范围，使其留出一点边距
#     if x_vals:
#         plt.xlim(0, 2**n_qubits)
        
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
    
#     # 保存图片并展示
#     save_path = 'counts_decimal_plot.png'
#     plt.savefig(save_path, dpi=300)
#     print(f"\n -> 绘制成功！柱状图已保存至: {save_path}")
    
#     plt.show() 

# else:
#     print("未找到解或未能生成 counts，无法绘图。")



if counts is not None:
    print(f"真实周期: {correct_period}")
    
    # ==========================================
    # 3. 数据处理: 仅保留十进制 >= 200 的态
    # ==========================================
    # 将字典的键转换为十进制，并过滤掉所有 < 200 的项（自然也就排除了 0）
    decimal_counts = {int(k, 2): v for k, v in counts.items() if int(k, 2) >= 200}
    
    # 按十进制数值大小对字典进行排序，保证画图时 X 轴从左到右递增
    sorted_counts = dict(sorted(decimal_counts.items()))
    
    # ==========================================
    # 4. 开始绘图
    # ==========================================
    plt.figure(figsize=(16, 6))
    
    x_vals = list(sorted_counts.keys())
    y_vals = list(sorted_counts.values())
    
    # 绘制柱状图
    plt.bar(x_vals, y_vals, color='royalblue', width=2.0)
    
    plt.title(f'Quantum Circuit Measurement Counts vs Decimal State\n(a={a}, c={c}, N={N}, n={n_qubits}) | States $>=$ 200 Only', fontsize=14)
    plt.xlabel('Measured State (Decimal)', fontsize=12)
    plt.ylabel(f'Counts (out of {shots} shots)', fontsize=12)
    
    # 设置 X 轴的范围，从 190 (留白) 到 2^n，聚焦目标区域
    if x_vals:
        plt.xlim(190, 2**n_qubits)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片并展示
    save_path = 'counts_decimal_filtered_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n -> 绘制成功！柱状图已保存至: {save_path}")
    
    plt.show() 

else:
    print("未找到解或未能生成 counts，无法绘图。")