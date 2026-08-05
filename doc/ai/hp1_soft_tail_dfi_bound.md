# HP-1 soft-tail 函数给出的直接 DFI 下界

## 1. 定义一个不需要相减两个大数的函数

令

\[
N=2^n,\qquad P(x)=P_r(x),\qquad Q(x)=P_{r+1}(x),
\]

并定义 scaled point probabilities

\[
p(x)=NP(x),\qquad q(x)=NQ(x).
\]

对固定常数 \(C>0\) 和 \(s>0\)，定义

\[
\boxed{
F_{s,C}(n,r)
:=
\frac1C\,
\mathbb E_{X\sim\operatorname{Unif}(\{0,1\}^n)}
\left[
(q(X)-p(X))^2
\left(1-\left(\frac{p(X)}C\right)^s\right)_+
\right].
}
\tag{1}
\]

式 (1) 是一个非负 expectation，不涉及两个接近的大数相减，也不需要
\(P(x)>0\)。

## 2. 为什么它严格 lower-bound exact DFI

记

\[
D=\sum_x(Q(x)-P(x))^2,
\qquad
B_C=\{x:p(x)>C\}.
\]

定义 bounded soft indicator

\[
\phi_{s,C}(p)=\min\left\{1,(p/C)^s\right\}.
\]

因为 \(\phi_{s,C}(p)=1\) on \(B_C\)，所以

\[
D(B_C)
\le
\sum_x(Q-P)^2\phi_{s,C}(p)
=:f_{s,C}(n,r)D,
\tag{2}
\]

其中 \(0\le f_{s,C}\le1\)。因此

\[
D-D(B_C)
\ge
(1-f_{s,C})D
=
\sum_x(Q-P)^2
\left(1-(p/C)^s\right)_+.
\tag{3}
\]

在 \(p<C\) 上，\(1/P=N/p>N/C\)。由式 (3)，

\[
\begin{aligned}
I_r
&=\sum_x\frac{(Q-P)^2}{P}\\
&\ge
\frac NC
\sum_x(Q-P)^2
\left(1-(p/C)^s\right)_+\\
&=F_{s,C}(n,r).
\end{aligned}
\tag{4}
\]

所以有无条件的点态下界

\[
\boxed{I_r\ge F_{s,C}(n,r).}
\tag{5}
\]

它不再需要另外假设 \(D\gtrsim r^{-2}\)、active-set relation 或
\(P(x)>0\)。\(s=1\) 是最简单的线性 ramp；\(s\to\infty\) 回到 hard
cutoff \(p<C\)。

## 3. 非 dyadic 严格窗口的 exact 数值

设置：

- fixed-phase HP-1，源代码 little-endian convention；
- exact support \(\{0,r,2r,\ldots\}\cap[0,N)\)；
- \(C=2\)、\(s=1\)；
- 严格窗口 \(2\le r<2^{n/4}\)；
- 排除所有 \(r=2^a\)；
- complex128 full-state calculation。

得到窗口内的最小 soft-tail bound：

| \(n\) | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(\min_r F_{1,2}\) | 4.869 | 5.921 | 9.059 | 11.272 | 14.576 | 21.313 | 31.619 | 38.519 | 58.441 |

在 \(n=14,\ldots,22\) 上，

\[
\boxed{
\min_{\substack{2\le r<2^{n/4}\\r\ne2^a}}
F_{1,2}(n,r)
\approx
\exp(0.31158n-2.84193),
\qquad R^2=0.99503.
}
\tag{6}
\]

改变 soft power 不改变结论：同一窗口上 \(s=2,4,8,\infty\) 的 fitted
slopes 分别约为 \(0.3063,0.3036,0.3022,0.3018\)。因此正 slope 不是
\(s=1\) 的特殊调参结果。

对式 (2) 的 normalized deficit，完整非 dyadic 扫描在
\(n=15,\ldots,22\) 给出

\[
\min_r(1-f_{1,2})
\approx
2.795\,e^{-0.32572n},
\qquad R^2=0.99944.
\tag{7}
\]

式 (6) 是直接 DFI lower bound 的有限尺寸数值证据；它不是渐近证明。

## 4. 不构造 statevector 的大比特估计

式 (1) 只需要均匀抽取输出 bitstring \(x\)，再 point-query \(P_r(x)\) 和
\(P_{r+1}(x)\)。roots-of-unity 公式把每个 point query 的代价降为

\[
O\bigl(n(u_r+u_{r+1})\bigr),
\qquad r=2^a u_r,
\]

且不使用 \(O(2^n)\) 内存。实现：

```bash
uv run python scripts/experiments/hp1_soft_tail_dfi_mc.py \
  --n 40 --periods 12,20 --threshold-c 2 --powers 1,2,4 \
  --sample-count 1000000
```

对应核心实现位于 `src/altqft/fi/small_probability.py` 的
`soft_tail_dfi_samples`。

需要诚实区分两种 scaling：

1. **每点计算和内存 scaling**：不再含 \(2^n\) statevector；当 odd parts
   为 fixed/poly\((n)\) 时是 polynomial。
2. **Monte Carlo 相对误差 scaling**：integrand 仍可能由稀有 resonance
   outputs 主导。脚本同时报告 standard error、ESS 和最大单点权重占比；
   ESS 太小时不能把 point estimate 当成可靠 scaling evidence。

### 4.1 原生 CUDA 实现与 n=300 smoke test

原生 CUDA 版本位于
`scripts/experiments/hp1_tail_dfi_mc_cuda.cu`。一个 warp 处理一个均匀输出
串，在 shared memory 中构造 HP-1 digit weights，并沿 roots-of-unity
frequencies 并行计算 \(p=NP_r(x)\) 和 \(q=NP_{r+1}(x)\)。它直接估计
hard-cutoff 量

\[
L_2(n,r)=\frac12\mathbb E_{X\sim U}
[(q(X)-p(X))^2\mathbf1\{p(X)<2\}]
\le I_r(n).
\]

编译和运行：

```bash
nvcc -O3 -std=c++17 -arch=native \
  scripts/experiments/hp1_tail_dfi_mc_cuda.cu \
  -o hp1_tail_dfi_mc_cuda

./hp1_tail_dfi_mc_cuda \
  --n-min 20 --n-max 300 --n-step 10 \
  --periods 12 --samples 1048576 \
  --output data/hp1_tail_dfi_cuda/r12.csv
```

在 RTX 5060 Ti 上，\(n=300,r=12\) 的 65,536-point smoke test 约需
1.3 秒；1,048,576 points 约需 6.4 秒。这证明 point-forward 和内存已经
可以扩展到 300 bits。但是 uniform sampling 在固定样本数下并没有统计
scaling：n=300 smoke test 的 ESS 约为 1，最大样本占总估计约 0.999。
所以该行被程序标为 `LOW_ESS`，不会进入 regression。换言之，CUDA
解决了 forward-compute scaling，但没有自动解决 rare-event sample
complexity。

若可以直接从量子电路抽取 \(P_r/P_{r+1}\) shots，则更好的 proposal 是
\(M=(P_r+P_{r+1})/2\)。在 scaled variables 中，式 (1) 等价于

\[
F_{s,C}
=
\mathbb E_{X\sim M}
\left[
\frac{2}{C}\frac{(q-p)^2}{p+q}
\left(1-(p/C)^s\right)_+
\right],
\tag{8}
\]

它会自动采到 numerator resonance outputs，适合真正的大比特 shot-based
benchmark。
