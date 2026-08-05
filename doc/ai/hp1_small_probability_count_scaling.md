# HP-1 指数小概率输出的可扩展计数

## 1. 要计算的量

令

\[
N=2^n,\qquad
\Omega_C(n,r):=\{x\in\{0,1\}^n:P_r(x)\le C/N\},
\]

并定义 uniform counting fraction

\[
a_C(n,r):=\frac{|\Omega_C(n,r)|}{N}.
\]

更一般地，阈值 \(P_r(x)\le 2^{-\beta n}\) 等价于

\[
NP_r(x)\le 2^{(1-\beta)n}.
\]

这里必须区分 \(|\Omega_C|/N\) 与概率质量
\(\sum_{x\in\Omega_C}P_r(x)\)：前者是均匀计数占比，不能用 HP-1
测量样本直接无偏估计。

## 2. 不做任何电路计算就有的严格下界

令 \(B_C=\Omega_C^c=\{x:P_r(x)>C/N\}\)。由归一化，

\[
1\ge \sum_{x\in B_C}P_r(x)>|B_C|\frac C N,
\]

所以

\[
\boxed{
\frac{|\Omega_C|}{N}>1-\frac1C
}
\qquad(C>1).
\]

例如 \(P_r(x)\le2/N\) 的输出点严格超过全部输出的一半；
\(P_r(x)\le4/N\) 的输出点严格超过四分之三。这个结论与 HP-1
结构无关。

若阈值是 \(2^{-\beta n}\) 且 \(\beta<1\)，则

\[
\frac{|\Omega_\beta|}{N}>1-2^{-(1-\beta)n}.
\]

因此这一区间内，占比本身已经趋于 1；真正需要数值计算的是临界尺度
\(P=\Theta(1/N)\)（即 \(\beta=1\)）附近的 CDF。

### 2.1 Dyadic period 有更强的精确零点计数

若 \(r=2^a\)，下面式 (1) 中的 odd part 是 \(u=1\)。每个满足
\(j\ge a\) 的偶数输入位贡献因子

\[
1+z_j(x)=1+(-1)^{x_j}.
\]

所以只要某个偶数输出位 \(j\ge a\) 满足 \(x_j=1\)，就有
\(P_{2^a}(x)=0\)。独立约束的数目为

\[
k_E=\left\lceil\frac n2\right\rceil-
\left\lceil\frac a2\right\rceil.
\]

因此对任意正阈值（不只 \(C/N\)），都有严格计数下界

\[
\boxed{
|\Omega|\ge 2^n-2^{n-k_E},
\qquad
\frac{|\Omega|}{2^n}\ge1-2^{-k_E}.
}
\]

例如 \(n=40,r=2\) 或 \(4\) 时 \(k_E=19\)，至少
\(1-2^{-19}\approx0.9999981\) 的输出具有精确零概率。这个解析下界比
uniform Monte Carlo 更适合识别如此稀少的补集。

## 3. 把单点概率从 \(O(2^n/r)\) 化为 \(O(nu)\)

采用源代码 `HPcore.py` 的 0-based little-endian qubit convention。令

\[
r=2^a u,\qquad u\text{ 为奇数}.
\]

对输出串 \(x\)，定义每个输入 bit 的权重

\[
z_j(x)=
\begin{cases}
(-1)^{x_j},&j\text{ even},\\
(-1)^{x_j}\exp\!\left(i\sum_{e\text{ even}}
\dfrac{\pi}{2^{|e-j|}}x_e\right),&j\text{ odd}.
\end{cases}
\]

HP-1 的 phase sum 是

\[
T_r(x)=\sum_{\substack{0\le y<N\\r\mid y}}
\prod_{j=0}^{n-1}z_j(x)^{y_j},
\qquad
P_r(x)=\frac{|T_r(x)|^2}{NR_r},
\quad R_r=\left\lceil\frac Nr\right\rceil.
\]

由于 \(r=2^a u\)，条件 \(r\mid y\) 强制低 \(a\) 位为零。写
\(y=2^a v\)，再对 \(u\mid v\) 使用 roots-of-unity filter，得到精确恒等式

\[
\boxed{
T_r(x)=\frac1u\sum_{h=0}^{u-1}
\prod_{j=a}^{n-1}
\left(1+z_j(x)e^{2\pi i h2^{j-a}/u}\right).
}
\tag{1}
\]

为避免大数，实际计算归一化量

\[
G_r(x):=\frac{T_r(x)}{2^{n-a}}
=\frac1u\sum_{h=0}^{u-1}
\prod_{j=a}^{n-1}
\frac{1+z_j(x)e^{2\pi i h2^{j-a}/u}}2.
\]

于是

\[
\boxed{
\log_2\!\bigl(NP_r(x)\bigr)
=2(n-a)-\log_2R_r+2\log_2|G_r(x)|.
}
\tag{2}
\]

式 (1) 把每个点的代价从直接相干求和的
\(O(R_r)=O(2^n/r)\) 降到 \(O((n-a)u)\)，内存可通过分块保持为
\(O(M_{\rm chunk}u_{\rm chunk})\)。尤其：

- dyadic period \(r=2^a\) 时 \(u=1\)，每点只需 \(O(n)\)；
- odd part \(u=\operatorname{poly}(n)\) 时，每点是多项式代价；
- 最坏窗口 \(r<2^{n/4}\) 中仍可能有 \(u=\Theta(2^{n/4})\)，所以这是
  sub-exponential 改进，不应宣称为对全部窗口都 polynomial。

## 4. 把 \(2^n\) 个输出的计数改成 uniform Monte Carlo

取 \(X\sim\operatorname{Unif}(\{0,1\}^n)\)，则

\[
\boxed{
a_C(n,r)
=\mathbb E_X\left[
\mathbf 1\{\log_2(NP_r(X))\le\log_2C\}
\right].
}
\tag{3}
\]

因此均匀抽取 \(M\) 个输出串，利用式 (2) 做 point query，即得到无偏估计

\[
\widehat a_C=\frac1M\sum_{m=1}^M
\mathbf 1\{NP_r(X_m)\le C\}.
\]

Hoeffding 不等式给出

\[
\Pr\bigl(|\widehat a_C-a_C|>\epsilon\bigr)
\le2e^{-2M\epsilon^2}.
\]

所以固定加性精度 \(\epsilon\) 和失败概率 \(\delta\) 只需

\[
M\ge\frac{\log(2/\delta)}{2\epsilon^2},
\]

样本数不依赖 \(2^n\)。总计算量为

\[
O\!\left(M(n-a)u\right),
\]

而不是构造完整分布所需的 \(O(2^n)\) 内存和时间。

仓库实现：

```bash
uv run python scripts/experiments/hp1_small_probability_fraction.py \
  --n 40 \
  --periods 2,4,12,20 \
  --c-values 1,2,4 \
  --sample-count 100000 \
  --output data/hp1_small_probability/n40.csv
```

也可直接指定指数阈值：

```bash
uv run python scripts/experiments/hp1_small_probability_fraction.py \
  --n 40 --periods 12 --c-values '' --beta-values 0.9,1.0,1.1
```

实现位于：

- `src/altqft/fi/small_probability.py`：式 (1)--(2) 的 complex128 point query；
- `scripts/experiments/hp1_small_probability_fraction.py`：式 (3) 的 uniform Monte Carlo 和置信区间；
- `tests/fi/test_small_probability.py`：与小规模 Qiskit statevector 逐点核对。

## 5. 能做与不能做的结论

这套变换可以可扩展地估计 \(|\Omega_C|/2^n\)，但不自动给出
\(\Omega_C\) 上的 numerator energy
\(\sum_{x\in\Omega_C}(P_{r+1}(x)-P_r(x))^2\)。后者仍需要同时 point-query
\(P_r,P_{r+1}\)，并且若要估计很稀有的 energy spike，uniform Monte Carlo
可能有大方差。

另外，从 HP-1 实验 shots（按 \(P_r\) 抽样）不能可靠恢复 uniform support
count：小概率点恰好会被欠采样。式 (3) 必须均匀抽 \(x\)，再计算其 point
probability。
