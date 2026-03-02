import sympy
from sympy import symbols, Matrix, I, exp, pi, S
from sympy.physics.quantum import TensorProduct

# 定义符号 theta
theta = symbols('theta')

# 定义基础矩阵
H = (S(1)/sympy.sqrt(2)) * Matrix([[1, 1], [1, -1]])
id2 = Matrix([[1, 0], [0, 1]])

# CP(theta) 矩阵：仅在 |11> 态增加相位 e^(i*theta)
CP = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, exp(I * theta)]
])

# 构造 H1 和 H2 (假设是 2-qubit 系统)
H1 = TensorProduct(H, id2)
H2 = TensorProduct(id2, H)

# 计算复合矩阵
final_matrix = H2 * CP * H1

# 打印结果
sympy.pprint(final_matrix)