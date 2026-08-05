# HP-1 偶数周期小分母分子能量下界：证明尝试

> 工作状态：部分结果；尚未闭合 Theorem 2。本文只记录可严格成立的归约、一个 exact-zero 子类，以及目前无法跨越的步骤。

## 1. 要证明的命题必须先归一化

令

\[
N=2^n,\qquad P(x)=P_r(x),\qquad Q(x)=P_{r+1}(x),
\qquad \Delta(x)=Q(x)-P(x),
\]

\[
D_r=\sum_x\Delta(x)^2.
\]

对常数或缓慢增长的函数 \(C\ge 2\)，定义小分母集合

\[
A_C=A_C(n,r):=\{x:P_r(x)\le C/N\},
\qquad a_C:=\frac{|A_C|}{N},
\]

以及该集合上的分子能量

\[
D_r(A_C):=\sum_{x\in A_C}\Delta(x)^2.
\]

如果 \(|A_C|\) 表示点数，则

\[
D_r(A_C)\ge k\,|A_C|D_r
\]

不可能以固定 \(k>0\) 对大集合成立，因为总有 \(D_r(A_C)\le D_r\)。正确的无量纲形式是

\[
\boxed{
D_r(A_C)\ge k_n\,a_C D_r.
}
\tag{T}
\]

常数版本对应 \(k_n\ge k>0\)。事实上，为闭合正指数下界，只需更弱的

\[
k_n\ge c_k e^{-\alpha n},
\qquad \alpha<\frac{\ln 2}{2}.
\tag{T'}
\]

## 2. (T) 为什么足以闭合偶数周期部分

首先，归一化立即给出小分母集合的计数下界。令

\[
B_C=\{x:P_r(x)>C/N\}.
\]

则

\[
1\ge\sum_{x\in B_C}P_r(x)>|B_C|C/N,
\]

所以

\[
|B_C|<N/C,
\qquad
\boxed{a_C>1-1/C.}
\tag{1}
\]

另一方面，E40 在 \(A_C\) 上给出

\[
I_r
\ge \sum_{x\in A_C}\frac{\Delta(x)^2}{P_r(x)}
\ge \frac{N}{C}D_r(A_C).
\tag{2}
\]

若 (T) 成立，则

\[
I_r
\ge \frac{N}{C}k_na_CD_r
\ge \frac{1-1/C}{C}k_nND_r.
\tag{3}
\]

因此，如果还能对所有 admissible \(r\) 严格证明

\[
D_r\ge c_D/r^2,
\tag{4}
\]

那么在 \(r<N^{1/4}\) 中

\[
I_r
\ge \frac{1-1/C}{C}c_Dk_n\frac{N}{r^2}
\ge c\,k_nN^{1/2}.
\tag{5}
\]

- 若 \(k_n\ge k>0\)，得到 \(I_r=\Omega(N^{1/2})\)；
- 若 (T') 成立，得到
  \[
  I_r\ge c\exp[(\ln2/2-\alpha)n],
  \]
  仍是正指数。

所以这个思路在逻辑上是正确的。不过，式 (4) 目前也只有统计 overlap law，尚没有覆盖所有 2-adic 类并带显式误差的统一证明。

## 3. 普通占比不能推出分子能量占比

式 (1) 只控制 uniform counting measure 下的占比。DFI 需要控制的是变化权重

\[
w_r(x):=\frac{\Delta(x)^2}{D_r}
\]

下的占比：

\[
\rho_C(n,r)
:=\sum_{x\in A_C}w_r(x)
=\frac{D_r(A_C)}{D_r}.
\tag{6}
\]

目标 (T) 正是

\[
\rho_C(n,r)\ge k_na_C.
\]

两种占比之间没有一般的确定性关系。例如，令 \(N-2\) 个点的 \(P\) 都等于 \(1/(2N)\)，剩余两个点平分其余概率；再令 \(Q-P\) 只在最后两个大概率点之间转移。此时

\[
|A_2|/N\to1,
\qquad
D(A_2)=0.
\]

因此，仅靠归一化、\(|A_C|\) 或三个二次 overlaps 不可能推出 (T)。必须使用 HP-1 的额外结构。

### 3.1 按 numerator 再挑子集只能把困难转移到集合大小

如果允许根据 numerator 自适应地选集合，可以定义

\[
A_{C,k}^{\mathrm{act}}
:=\left\{x:
P_r(x)\le C/N,\quad
\Delta(x)^2\ge kD_r/N
\right\}.
\]

那么按照定义立刻有

\[
D_r(A_{C,k}^{\mathrm{act}})
\ge k\frac{|A_{C,k}^{\mathrm{act}}|}{N}D_r.
\]

所以若只要求“存在某个集合使常数 \(k\) 关系成立”，这个命题可以通过定义集合而平凡地满足。真正还需要证明的是

\[
\frac{|A_{C,k}^{\mathrm{act}}|}{N}
\ge c_Ae^{-\beta n}
\]

且指数必须满足

\[
\alpha+\beta<\ln2/2
\]

（常数 \(k\) 时 \(\alpha=0\)）。否则集合可能只有一个点，式 (8) 仍不足以推出正指数 DFI。换言之，“能挑出 active 小分母项”不是难点；难点是证明这种项统一地足够多。

## 4. 一个精确的 moment 归约

令

\[
H_r:=\sum_xP_r(x)\Delta(x)^2.
\tag{7}
\]

在补集 \(B_C\) 上有 \(P_r(x)>C/N\)，故

\[
H_r
\ge\sum_{x\in B_C}P_r(x)\Delta(x)^2
>\frac{C}{N}\sum_{x\in B_C}\Delta(x)^2.
\]

因此

\[
D_r(B_C)<\frac{N}{C}H_r,
\qquad
\boxed{
D_r(A_C)>D_r-\frac{N}{C}H_r.
}
\tag{8}
\]

而

\[
H_r
=\sum_xP_rQ_r^2
 -2\sum_xP_r^2Q_r
 +\sum_xP_r^3.
\tag{9}
\]

这说明一种可能的证明路线是估计三次 mixed overlaps。若能证明

\[
H_r
\le \frac{C}{N}(1-k_na_C)D_r,
\tag{10}
\]

则 (8) 立即给出 (T)。当前附录中的三个二次 overlaps 不包含式 (9) 的信息，因此不能完成这一步。

## 5. 一个更强但清楚的四阶充分条件

记

\[
F_r:=\sum_x\Delta(x)^4.
\]

由 Cauchy--Schwarz 和 \(|B_C|<N/C\)，

\[
D_r(B_C)
\le\sqrt{|B_C|F_r}
<\sqrt{\frac{N}{C}F_r}.
\tag{11}
\]

若能证明 numerator delocalization

\[
F_r\le K_n\frac{D_r^2}{N},
\tag{12}
\]

则

\[
D_r(B_C)\le D_r\sqrt{K_n/C}.
\]

取 \(C=4K_n\) 可得

\[
D_r(A_C)\ge D_r/2,
\qquad
I_r\ge \frac{N}{8K_n}D_r.
\tag{13}
\]

如果 \(K_n=\operatorname{poly}(n)\)，式 (4) 与 (13) 会给出

\[
I_r\ge \frac{N^{1/2}}{\operatorname{poly}(n)}.
\]

这是一个完整的条件证明模板。但对偶数周期，numerator 能量可能集中在少数 resonance outputs；当前没有式 (12) 的统一证明，数值中的全局 \(K_n\) 也很大。因此 (12) 是充分条件，不是已经验证的 HP-1 性质。

## 6. HP-1 的精确 phase-sum 表示

采用 `src/altqft/circuits/HPcore.py` 和 Qiskit state index 的 0-based little-endian convention，令

\[
E=\{0,2,4,\ldots\},
\qquad
O=\{1,3,5,\ldots\}.
\]

需要注意：`scripts/experiments/hp1_chernoff_window_exact.py` 的 docstring 和实现采用“qubit 0 is most-significant bit”。当 \(n\) 为偶数时，bit reversal 会交换整数位上的奇偶 partition，所以该脚本与这里的源代码口径不是自动等价的。把本节结论并入论文前，必须先统一 convention 并重跑相应证书。

由 `HPcore.py` 可直接得到矩阵元

\[
U_{x,y}
=N^{-1/2}
(-1)^{x_E\cdot y_E+x_O\cdot y_O}
\exp\!\left(
 i\sum_{e\in E}\sum_{o\in O}
 \frac{\pi}{2^{|e-o|}}x_ey_o
\right).
\tag{14}
\]

令

\[
R_s=\left\lceil\frac Ns\right\rceil,
\qquad
T_s(x)=\sum_{q=0}^{R_s-1}\sqrt N\,U_{x,qs}.
\]

则

\[
P_s(x)=\frac{|T_s(x)|^2}{NR_s}.
\tag{15}
\]

所以 \(A_C\) 等价于

\[
|T_r(x)|^2\le CR_r,
\tag{16}
\]

且

\[
D_r(A_C)
=\frac1{N^2}
\sum_{|T_r(x)|^2\le CR_r}
\left(
 \frac{|T_{r+1}(x)|^2}{R_{r+1}}
 -\frac{|T_r(x)|^2}{R_r}
\right)^2.
\tag{17}
\]

因此目标 (T) 是一个关于两组 dyadic root-of-unity sums 的加权 small-ball estimate。普通 Parseval 恒等式只给出总和，不能说明式 (17) 中的能量是否落在 \(|T_r|^2\le CR_r\) 的部分。

### 6.1 一个绕开 controlled phase 的精确边缘归约

还有一个比式 (17) 更简单的严格归约。把输出位分成偶数位
\(E=\{0,2,\ldots\}\) 和奇数位 \(O=\{1,3,\ldots\}\)，并记

\[
m_E=|E|=\lceil n/2\rceil,
\qquad
m_O=|O|=\lfloor n/2\rfloor,
\qquad
M_E=2^{m_E}.
\]

对紧凑位串 \(e\in\{0,1\}^{m_E}\)、\(o\in\{0,1\}^{m_O}\)，定义 little-endian 交错映射

\[
\iota(e,o)
:=
\sum_{k=0}^{m_E-1}e_k2^{2k}
+
\sum_{k=0}^{m_O-1}o_k2^{2k+1}.
\tag{18a}
\]

令 \(P_s^E(\xi)\) 表示最终 HP-1 输出分布在偶数输出位上的边缘。HP-1 先在偶数位施加 Hadamard，随后施加的 controlled-phase 是对角门，而最后的 Hadamard 只作用于奇数位。因此，后两步都不改变偶数位的测量边缘。直接在 controlled-phase 之前计算可得精确公式

\[
\boxed{
P_s^E(\xi)
=
\frac{1}{M_ER_s}
\sum_{o\in\{0,1\}^{m_O}}
\left|
\sum_{\substack{e\in\{0,1\}^{m_E}\\
s\mid\iota(e,o)}}
(-1)^{\xi\cdot e}
\right|^2 .
}
\tag{18b}
\]

特别地，式 (18b) 的内层和是整数；这个边缘问题已经完全不含 HP-1 的 dyadic controlled phases。

这个公式也可写成 root-of-unity product。令
\(\zeta_s=e^{2\pi i/s}\) 以及
\(\iota_O(o)=\sum_k o_k2^{2k+1}\)，则整除指示函数给出

\[
\sum_{\substack{e\in\{0,1\}^{m_E}\\s\mid\iota(e,o)}}
(-1)^{\xi\cdot e}
=
\frac1s\sum_{h=0}^{s-1}
\zeta_s^{h\iota_O(o)}
\prod_{k=0}^{m_E-1}
\left(1+(-1)^{\xi_k}\zeta_s^{h4^k}\right).
\tag{18c}
\]

若记

\[
K_s(\xi)
:=
\sum_o
\left|
\sum_{\substack{e\\s\mid\iota(e,o)}}
(-1)^{\xi\cdot e}
\right|^2
\in\mathbb Z_{\ge0},
\]

则 \(P_s^E(\xi)=K_s(\xi)/(M_ER_s)\)，并且
\(\sum_\xi K_s(\xi)=M_ER_s\)。在不存在边缘 support mismatch 时，还得到完全整数化的恒等式

\[
\boxed{
I_r^E
=
\frac{1}{M_E R_rR_{r+1}^2}
\sum_{\xi:K_r(\xi)>0}
\frac{
\left(R_rK_{r+1}(\xi)-R_{r+1}K_r(\xi)\right)^2
}{K_r(\xi)}.
}
\tag{18c'}
\]

若某个 \(\xi\) 满足 \(K_r(\xi)=0<K_{r+1}(\xi)\)，则直接有 \(I_r^E=+\infty\)。所以非零有限情形真正缺少的是式 (18c') 中这个整数加权和的统一指数下界，而不是 HP-1 controlled phases 的估计。

令

\[
I_r^E
:=
\sum_{\xi}
\frac{\left(P_{r+1}^E(\xi)-P_r^E(\xi)\right)^2}
{P_r^E(\xi)},
\tag{18d}
\]

并继续采用 extended-value 零分母约定。对每个固定 \(\xi\)，在所有奇数输出位上应用加权 Cauchy--Schwarz，得到

\[
\sum_{x_O}
\frac{\left(P_{r+1}(\xi,x_O)-P_r(\xi,x_O)\right)^2}
{P_r(\xi,x_O)}
\ge
\frac{\left(P_{r+1}^E(\xi)-P_r^E(\xi)\right)^2}
{P_r^E(\xi)}.
\]

求和后即为 Pearson \(\chi^2\) 的 data-processing inequality：

\[
\boxed{I_r\ge I_r^E.}
\tag{18e}
\]

因此，可以完全绕开 (T)、三次 mixed moments 和四次 delocalization，转而证明

\[
\min_{\substack{2\le r<2^{n/4}\\r\text{ even}}} I_r^E
\ge c e^{\beta n},
\qquad \beta>0.
\tag{18f}
\]

式 (18b)--(18c) 把这个新目标化成了一个模 \(r\) 与模 \(r+1\) 的整数 Walsh-product \(\chi^2\) 下界。它比原 phase-sum tail 少了一层 HP-1 相位结构，但目前仍没有对所有非 dyadic 偶数 \(r\) 的统一解析下界。

## 7. 可严格证明的偶数周期 exact-zero 子类

下面的子类可以完全处理，并且不需要 (T)。

令

\[
r=2^au,
\qquad a=\nu_2(r)\ge1,
\qquad u\text{ 为奇数}.
\]

取任意 odd target index \(j\ge a\)，并令输出 \(x\) 只在第 \(j\) 位等于 1。此时 \(x_E=0\)，所有 controlled-phase 因子消失，因此

\[
T_r(x)
=\sum_{q=0}^{R_r-1}(-1)^{(qr)_j}
=\sum_{q=0}^{R_r-1}(-1)^{(qu)_{j-a}}.
\tag{18}
\]

该符号序列的周期为

\[
L=2^{j-a+1}
\]

且每个完整周期内正负项数相同。因此：

**Exact-zero 引理。** 若

\[
2^{j-a+1}\mid R_r,
\tag{19}
\]

则 \(T_r(x)=0\)，即 \(P_r(x)=0\)。若进一步 \(R_{r+1}\) 为奇数，则

\[
T_{r+1}(x)=
\sum_{q=0}^{R_{r+1}-1}(-1)^{(q(r+1))_j}\ne0,
\tag{20}
\]

因为它是奇数个 \(\pm1\) 的和。于是

\[
P_r(x)=0<P_{r+1}(x),
\qquad
I_r=+\infty.
\tag{21}
\]

这严格覆盖了一部分偶数周期和 2-adic 类，但条件 (19)--(20) 并非对所有 admissible \(n,r\) 成立。

### 7.1 全部 dyadic 周期 \(r=2^a\)

上面的单 target-bit 引理并不是处理二进制幂周期的最强方法。事实上，可以用偶数位边缘一次覆盖窗口内所有 \(r=2^a\)。

**Dyadic support-mismatch 引理。** 设

\[
r=2^a,
\qquad
2\le r<2^{n/4}.
\]

则存在输出 \(x\) 使

\[
P_r(x)=0<P_{r+1}(x),
\]

因而 exact DFI \(I_r=+\infty\)。

**证明。** 精确周期态因子化为

\[
|\psi_{2^a}\rangle
=
|0\rangle_0\cdots|0\rangle_{a-1}
\bigotimes_{j=a}^{n-1}|+\rangle_j.
\tag{21a}
\]

所以在第一层偶数位 Hadamard 之后，每个满足 \(e\in E\) 且 \(e\ge a\) 的偶数输出位都确定为 \(x_e=0\)。后续对角 controlled-phase 和奇数位 Hadamard 不改变偶数位边缘。因此 \(P_{2^a}^E\) 支持在真子空间

\[
V_a:=\{\xi:\xi_e=0\text{ for every even }e\ge a\}.
\tag{21b}
\]

窗口条件 \(a<n/4\) 保证至少存在一个偶数位 \(e\ge a\) 且 \(e<n\)。令 \(t=2^a+1\)。在输入态 \(|\psi_t\rangle\) 上有

\[
\langle\psi_t|X_e|\psi_t\rangle=0.
\tag{21c}
\]

这是因为 \(|\psi_t\rangle\) 的 computational support 为

\[
S_t=\{0,t,2t,\ldots\}\cap[0,N),
\]

而 \(X_e\) 只连接相差 \(2^e\) 的两个 basis states；若两个状态都在 \(S_t\) 中，就会要求奇数 \(t>1\) 整除 \(2^e\)，不可能。于是对第 \(e\) 位施加 Hadamard 后，测得 \(x_e=1\) 的概率恰为

\[
\Pr_t(x_e=1)
=
\frac{1-\langle X_e\rangle_t}{2}
=rac12.
\tag{21d}
\]

因此 \(P_t^E(V_a^c)\ge1/2\)。故存在 \(\xi\notin V_a\) 满足
\(P_{2^a}^E(\xi)=0<P_t^E(\xi)\)。展开奇数位边缘后，至少存在一个完整输出 \(x=(\xi,x_O)\) 具有同样的 support mismatch。证毕。

这特别补上了原先 \(r=2\)、偶数 \(n\) 未覆盖的情形。周期 2 输入态为

\[
|\psi_2\rangle
=|0\rangle_0\bigotimes_{j=1}^{n-1}|+\rangle_j,
\]

而且 \(P_2\) 至多有 \(1+2^{\lfloor n/2\rfloor}\) 个非零完整输出；不过，对 \(I_2=+\infty\) 而言，上面的边缘 support 论证已经足够。

## 8. 数值检查对常数 \(k\) 的结论

使用 exact support 和严格窗口

\[
2\le r<2^{n/4},
\qquad n=10,\ldots,26,
\]

共扫描 512 个相邻周期对。取 \(C=2\)，并定义

\[
k_{\mathrm{eff}}(n,r)
:=\frac{D_r(A_2)}{a_2D_r}.
\tag{22}
\]

偶数周期的最小值为：

| \(n\) | 10 | 14 | 18 | 22 | 26 |
|---:|---:|---:|---:|---:|---:|
| \(\min_{r\text{ even}}k_{\mathrm{eff}}\) | \(3.999\times10^{-2}\) | \(1.089\times10^{-2}\) | \(3.296\times10^{-3}\) | \(1.051\times10^{-3}\) | \(3.432\times10^{-4}\) |

有限窗口拟合为

\[
\min_{r\text{ even}}k_{\mathrm{eff}}(n,r)
\approx e^{-0.2938n-0.3266},
\qquad R^2\approx0.9976.
\tag{23}
\]

该最小值在本窗口一直由 \(r=2\) 给出。若只看数值上没有检测到 zero mismatch 的偶数 pair，则有限窗口中的最小值约为

\[
1.71\times10^{-3}
\]

（出现在 \((n,r)=(23,12)\)），但其 lower envelope 仍随 \(n\) 下降；这不能证明存在与 \(n\) 无关的常数 \(k\)。代表点已用 complex128 复核：

\[
(n,r)=(23,12):
\quad
k_{\mathrm{eff}}\approx1.7117\times10^{-3},
\]

\[
(n,r)=(26,2):
\quad
k_{\mathrm{eff}}\approx3.4323\times10^{-4}.
\]

所以，当前数值不支持对全部偶数周期直接猜测常数版本 (T)。数值支持的是较弱的 (T')，其中

\[
\alpha\approx0.294<\ln2/2\approx0.347.
\]

如果这一 lower envelope 能被严格证明，再结合式 (4)，它仍足以产生约 \(e^{0.053n}\) 的正指数下界。

### 8.1 按 numerator 阈值挑 active 子集的实验

为直接测试第 3.1 节的方案，取 \(C=2\) 和固定常数

\[
k_0=10^{-4},
\]

并定义

\[
A^{\mathrm{act}}_{n,r}
=\left\{x:
P_r(x)\le2/N,
\quad
\Delta_r(x)^2\ge k_0D_r/N
\right\}.
\]

按照定义，常数关系

\[
D_r(A^{\mathrm{act}}_{n,r})
\ge k_0\frac{|A^{\mathrm{act}}_{n,r}|}{N}D_r
\]

自动成立。需要检查的只剩 active 集合是否足够大。对每个 \(n\) 在所有偶数 admissible \(r\) 中取最小 active fraction，得到：

| \(n\) | 10 | 14 | 18 | 22 | 26 |
|---:|---:|---:|---:|---:|---:|
| \(\min_{r\text{ even}}|A^{\mathrm{act}}|/N\) | \(4.023\times10^{-1}\) | \(1.508\times10^{-1}\) | \(4.562\times10^{-2}\) | \(1.088\times10^{-2}\) | \(2.437\times10^{-3}\) |

其中 \((n,r)=(22,4),(24,4),(26,4)\) 的 active counts 和 energy fractions 已用 complex128 逐点复核，与扫描值一致。

其有限窗口拟合为

\[
\min_{r\text{ even}}
\frac{|A^{\mathrm{act}}_{n,r}|}{N}
\approx e^{-0.32687n+2.6681},
\qquad R^2\approx0.9932.
\]

由于

\[
0.32687<\ln2/2\approx0.34657,
\]

如果这一 active-count lower envelope 能被解析证明，则式 (4) 会给出一个很弱但为正的指数

\[
I_r\gtrsim e^{(\ln2/2-0.32687)n}
\approx e^{0.0197n}.
\]

这与提出的“常数 \(k\) 乘以集合占比”形式完全一致。但是常数关系在这里是由集合定义保证的；**尚未证明的是 active 集合的指数级计数下界**。回归不能替代该证明。

存在一个精确的二阶矩归约。令

\[
z_x:=\frac{N\Delta_r(x)^2}{D_r},
\quad
L:=\{x:P_r(x)\le2/N\},
\]

\[
\rho_L:=\frac{D_r(L)}{D_r},
\qquad
K_L:=\frac{N\sum_{x\in L}\Delta_r(x)^4}{D_r^2}.
\]

对 uniform output measure，非 active 的低分母点满足 \(z_x<k_0\)，而 active 部分可用 Cauchy--Schwarz 控制，因此

\[
\rho_L
\le k_0+
\sqrt{
\frac{|A^{\mathrm{act}}|}{N}K_L
}.
\]

从而

\[
\boxed{
\frac{|A^{\mathrm{act}}|}{N}
\ge
\frac{(\rho_L-k_0)_+^2}{K_L}.
}
\]

所以 active-set 路线最终可归约为：证明低分母区域的能量 \(\rho_L\) 不过快消失，同时证明 restricted fourth moment \(K_L\) 不过快增长。当前没有这两个量的统一 HP-1 解析估计。

## 9. 当前结论

1. **闭合逻辑成立。** (T) 或较弱的 (T') 与统一的 \(D_r\ge c_D/r^2\) 足以处理偶数周期。
2. **常数 \(k\) 尚未证明。** 普通计数占比和二次 overlaps 均不足；完整偶数数值也不支持一个明显稳定的常数 lower envelope。
3. **exact-zero 覆盖有所扩大。** 条件 (19)--(20) 下 exact DFI 为 \(+\infty\)；此外，窗口内全部 dyadic 周期 \(r=2^a\) 都由第 7.1 节的 support-mismatch 引理覆盖。
4. **剩余核心仍是非 dyadic 偶数周期。** 原路线可攻击三次 moment (9)、四次 delocalization (12) 或 phase-sum tail (17)；第 6.1 节还给出一个不含 controlled phase 的替代目标，即直接证明整数 modular-Walsh 边缘量 \(I_r^E\) 正指数增长。
5. **Theorem 2 仍未闭合。** 对原 weighted-tail 路线，碰撞距离下界 (4) 仍须覆盖全部 2-adic 类并带显式误差；对边缘路线，则须建立式 (18f) 对所有非 dyadic 偶数周期的统一下界。
6. **本文还只处理了偶数 \(r\)。** 如果 Theorem 2 的 minimum 同时包含奇数 \(r\)，还需对奇数分母方向另作证明；由于 Pearson DFI 的分母固定为 \(P_r\)，不能把 \((r,r+1)\) 简单交换后沿用偶数结论。
7. **exact 与 fixed-cutoff DFI 必须分开。** 第 7.1 节证明的是 extended-value exact DFI。若数值量定义为 \(I_{r,\epsilon}=\sum_x\Delta_r(x)^2/\max(P_r(x),\epsilon)\) 且 \(\epsilon>0\) 固定，则总有 \(I_{r,\epsilon}\le2/\epsilon\)，不可能对 \(n\to\infty\) 保持无界的正指数下界。渐近定理必须明确采用 exact DFI，或规定随 \(n\) 衰减的 \(\epsilon_n\)。

一个适合下一步尝试的精确命题是以下 dichotomy：对每个偶数 admissible \(r\)，要么存在 \(x\) 使

\[
P_r(x)=0<P_{r+1}(x),
\]

要么存在统一的 \(c,\alpha>0\)，其中 \(\alpha<\ln2/2\)，使

\[
D_r(A_2)
\ge c e^{-\alpha n}a_2D_r.
\]

目前 exact-zero 引理只覆盖该 dichotomy 的一部分；第二支尚无解析证明。
