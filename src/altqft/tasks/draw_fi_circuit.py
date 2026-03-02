import matplotlib.pyplot as plt
import re
import os

def plot_loss_landscape(file_path: str, save_path: str = "loss_landscape.png"):
    """
    读取训练日志文件，并绘制横轴为 Epoch、纵轴为 Loss 的 Loss Landscape 折线图。
    
    参数:
        file_path (str): 训练日志 txt 文件的路径 (如 'training_landscape.txt')
        save_path (str): 输出图片的保存路径
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    # 用于存放解析出的数据
    # 数据结构: {2: {'epochs': [1, 2, ...], 'losses': [0.5, 0.4, ...]}, 4: {...}}
    data_dict = {}
    current_qubits = None

    # 定义正则表达式来提取信息
    # 匹配标识: "========== 开始训练 2 比特线路 =========="
    qubit_pattern = re.compile(r"开始训练\s+(\d+)\s+比特线路")
    # 匹配标识: "Epoch   1 | Loss: 1.386294 | Params: [0.0000]"
    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d\.\-]+)")

    # 1. 解析文件
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 检查是否是新的比特配置
            q_match = qubit_pattern.search(line)
            if q_match:
                current_qubits = int(q_match.group(1))
                data_dict[current_qubits] = {'epochs': [], 'losses': []}
                continue
            
            # 提取 Epoch 和 Loss
            if current_qubits is not None:
                e_match = epoch_pattern.search(line)
                if e_match:
                    epoch = int(e_match.group(1))
                    loss = float(e_match.group(2))
                    data_dict[current_qubits]['epochs'].append(epoch)
                    data_dict[current_qubits]['losses'].append(loss)

    if not data_dict:
        print("❌ 错误: 未能在文件中解析到任何训练数据，请检查文件内容格式。")
        return

    # 2. 开始绘图
    plt.figure(figsize=(10, 6))
    
    # 遍历不同的比特数，绘制多条折线
    # 按照比特数从小到大排序绘制
    for qubits in sorted(data_dict.keys()):
        epochs = data_dict[qubits]['epochs']
        losses = data_dict[qubits]['losses']
        
        if epochs and losses:
            plt.plot(epochs, losses, marker='o', markersize=3, linewidth=1.5, 
                     label=f'{qubits} Qubits')

    # 3. 设置图表样式
    plt.title('Loss Landscape vs Epochs for Quantum Circuits', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 设置网格与图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Circuit Scale", loc='upper right')
    
    # 紧凑布局以防标签被遮挡
    plt.tight_layout()
    
    # 保存并提示
    plt.savefig(save_path, dpi=300)
    print(f"✅ 绘图成功！图表已保存至: {save_path}")
    
    # 如果你在带有图形界面的本地机器上，取消注释下面这行可以直接弹出窗口预览
    # plt.show()

if __name__ == "__main__":
    # 调用函数，传入你保存的文件名
    # 你可以根据实际情况修改这个路径
    log_file = "data/training_landscape.txt"
    plot_loss_landscape(log_file)