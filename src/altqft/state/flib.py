def find_solutions(a, c, N, n):
    """
    找出所有满足 a^x ≡ c (mod N) 的 x (0 ≤ x < 2^n)
    
    参数:
        a: 底数
        c: 目标值 (0 ≤ c < N)
        N: 模数 (N ≥ 1)
        n: 量子比特数量 (x 的二进制表示长度)
    
    返回:
        solutions: 满足条件的 x 的列表 (未归一化)
    """
    # 处理 N=1 的特殊情况
    if N == 1:
        if c != 0:
            raise ValueError("c must be 0 when N=1")
        return list(range(1 << n))  # 所有 x 都满足条件
    
    total = 1 << n  # 2**n
    solutions = []
    
    for x in range(total):
        # 处理 a=0 和 x=0 的边界情况 (0^0 = 1)
        if x == 0:
            val = 1
        else:
            if a == 0:
                val = 0
            else:
                val = pow(a, x, N)
        
        # 检查条件
        if val == c:
            solutions.append(x)
    
    if not solutions:
        raise ValueError(f"No solution found for a={a}, c={c}, N={N}")
    
    return solutions