# def run_lr_on_initial_state(a, c, N, n, shots=1024):
#     """
#     执行完整流程：查找解 -> 准备初始态 -> 施加 LR 电路 -> 测量
    
#     参数:
#         a: 底数
#         c: 目标值
#         N: 模数
#         n: 量子比特数量
#         shots: 采样/测量次数 (默认: 1024)
#     """
#     # 1. 寻找满足条件的 x
#     try:
#         solutions = find_solutions(a, c, N, n)
#         print(f"找到的解集 solutions: {solutions}")
#     except ValueError as e:
#         print(f"错误: {e}")
#         return None, None
    
#     correct_period = solutions[1] - solutions[0] 

#     # 2. 生成初始态电路
#     qc_init = qiskit_initial_state(solutions, n)

#     # 3. 生成 LR 电路
#     lr_circ = lr_circuit_qiskit(n)

#     # 4. 组合电路：在初始态之后加上 LR 电路
#     complete_circuit = qc_init.compose(lr_circ)

#     # 5. 添加测量层 (自动生成名为 'meas' 的经典寄存器)
#     complete_circuit.measure_all()

#     # 6. 初始化 StatevectorSampler (Qiskit V2 原语)
#     sampler = StatevectorSampler()
    
#     # 7. 运行电路，传入自定义的 shots 参数
#     job = sampler.run([complete_circuit], shots=shots)
#     result = job.result()
    
#     # 8. 获取测量计数值
#     counts = result[0].data.meas.get_counts()

#     return complete_circuit, counts, correct_period