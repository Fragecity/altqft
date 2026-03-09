import unittest
import numpy as np
from qiskit.quantum_info import Statevector

from altqft.state.flib import find_solutions
from altqft.state.state import qiskit_initial_state

class TestFindSolutions(unittest.TestCase):
    
    def test_general_case(self):
        """测试一般情况：a=2, c=1, N=3, n=2 -> [0, 2]"""
        self.assertEqual(find_solutions(2, 1, 3, 2), [0, 2])
    
    def test_n1_case(self):
        """测试 N=1 的情况：c=0 时应返回所有 x"""
        self.assertEqual(find_solutions(5, 0, 1, 3), list(range(8)))  # 2^3=8
    
    def test_n1_invalid_c(self):
        """测试 N=1 但 c≠0 的情况：应报错"""
        with self.assertRaises(ValueError):
            find_solutions(5, 1, 1, 3)
    
    def test_a0_x0(self):
        """测试 a=0, x=0 的边界情况：0^0=1"""
        # c=1 时应包含 x=0
        self.assertEqual(find_solutions(0, 1, 5, 2), [0])
        # c=0 时不应包含 x=0
        self.assertEqual(find_solutions(0, 0, 5, 2), [1, 2, 3])
    
    def test_a0_x_positive(self):
        """测试 a=0, x>0 的情况：0^x=0"""
        # c=0 时应包含所有 x>0
        self.assertEqual(find_solutions(0, 0, 5, 3), list(range(1, 8)))
        # c=1 时不应包含 x>0
        self.assertEqual(find_solutions(0, 1, 5, 3), [0])
    
    def test_no_solution_corrected(self):
        """修正测试：无解的情况"""
        with self.assertRaises(ValueError):
            find_solutions(2, 3, 4, 2)  # 2^x mod 4: 1,2,0,0... 没有等于3的x
    
    def test_small_n(self):
        """测试 n=0 的边界情况（2^0=1，x=0）"""
        # a=2, c=1, N=3, n=0 -> x=0: 2^0 mod3=1 mod3=1 → 应包含
        self.assertEqual(find_solutions(2, 1, 3, 0), [0])
        # a=2, c=2, N=3, n=0 -> 2^0=1 ≠2 → 无解
        with self.assertRaises(ValueError):
            find_solutions(2, 2, 3, 0)
    
    def test_large_n(self):
        """测试大 n 的情况（n=10，2^10=1024）"""
        # a=2, c=1, N=1000000007（大质数）→ x=0 是解（2^0=1）
        solutions = find_solutions(2, 1, 1000000007, 10)
        self.assertEqual(solutions, [0])
    
    def test_periodic_solution(self):
        """测试周期性解（a=3, c=2, N=7, n=3）→ 3^2=9 mod7=2"""
        self.assertEqual(find_solutions(3, 2, 7, 3), [2])
    
    def test_mycase(self):
        solutions = find_solutions(13, 4, 111, 10)
        self.assertEqual(solutions, [10, 46, 82, 118, 154, 190, 226, 262, 298, 334, 370, 406, 442, 478, 514, 550, 586, 622, 658, 694, 730, 766, 802, 838, 874, 910, 946, 982, 1018])


class TestQiskitInitialState(unittest.TestCase):
    
    def test_state_fidelity(self):
        """
        使用 Statevector 验证电路生成的态是否正确。
        不再依赖 Aer 模拟器和测量，直接验证波函数。
        """
        solutions = [0, 2]  # 二进制对应 |00> 和 |10>
        n = 2
        
        # 1. 生成电路
        qc = qiskit_initial_state(solutions, n)
        
        # 2. 使用最新的 Statevector 接口获取末态
        state = Statevector.from_instruction(qc)
        
        # 3. 构造预期的理论态矢量
        expected_amplitude = 1 / np.sqrt(len(solutions))
        expected_state = np.zeros(2**n, dtype=complex)
        for idx in solutions:
            expected_state[idx] = expected_amplitude
        
        # 4. 验证保真度 (Fidelity) 或 直接对比
        is_match = np.allclose(state.data, expected_state)
        print(f"  -> 态矢量是否一致? {is_match}")
        
        self.assertTrue(is_match, f"生成的量子态与预期不符！预期: {expected_state}, 实际: {state.data}")

    def test_sample_counts(self):
        """
        模拟测量行为，验证采样结果是否只包含 solutions 中的索引。
        """
        solutions = [0, 2, 4, 6]
        n = 3
        
        print("\n--- [DEBUG] test_sample_counts 开始 ---")
        print(f"  预期的有效解 (solutions): {solutions}")
        print(f"  量子比特数 (n): {n}")
        
        qc = qiskit_initial_state(solutions, n)
        
        # 使用 Statevector 的 sample_counts 模拟 1000 次采样
        state = Statevector.from_instruction(qc)
        counts = state.sample_counts(shots=1000)
        
        print(f"  -> 原始采样统计 (二进制 counts): {counts}")
        
        # 验证采样出的 key（Qiskit 返回的是二进制字符串）是否在 solutions 中
        for bin_key, count in counts.items():
            decimal_key = int(bin_key, 2)
            print(f"  -> 解析测量值: 2进制 '{bin_key}' -> 10进制 {decimal_key} (共采样到 {count} 次)")
            
            self.assertIn(decimal_key, solutions, f"非法测量结果: {decimal_key} 不在预期解 {solutions} 中")
            
        print("--- [DEBUG] test_sample_counts 通过 ---")

if __name__ == '__main__':
    unittest.main()