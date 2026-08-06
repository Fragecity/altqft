# HP-1 active 小分母计数证书：可扩展到 200 比特的统计闭合

> 状态：确定性归约 + 明确标注的统计规律。本文只处理偶数非 dyadic 周期；dyadic
> 周期已有 exact support-mismatch 证明。奇数分母方向仍需单独处理。本文没有修改论文。

## 1. 不再把 active threshold 写成未知的全局量 \(D_r\)

令

\[
N=2^n,\qquad
P(x)=P_r(x),\qquad Q(x)=P_{r+1}(x),
\]

\[
p(x)=NP(x),\qquad q(x)=NQ(x),\qquad
\Delta(x)=Q(x)-P(x).
\]

固定一个与 \(n,r\) 无关的小常数 \(\tau>0\)。定义

\[
\boxed{
A_\tau(n,r)
=
\left\{x:
p(x)<2,\quad
(q(x)-p(x))^2r^2\ge \tau N
\right\}.
}
\tag{1}
\]

本轮数值固定

\[
\tau=3\times10^{-4}.
\tag{2}
\]

式 (1) 比旧定义
\(\Delta(x)^2\ge k_0D_r/N\) 更干净：它不依赖未知的 \(D_r\)，也不需要先使用
period-overlap law 推出 \(D_r\gtrsim r^{-2}\)。

## 2. 每个 active 点给出严格的常数贡献

若 \(x\in A_\tau(n,r)\)，则

\[
P(x)<\frac2N
\tag{3}
\]

且

\[
N^2\Delta(x)^2r^2
=(q(x)-p(x))^2r^2
\ge\tau N.
\]

所以

\[
\Delta(x)^2\ge\frac{\tau}{Nr^2}.
\tag{4}
\]

结合式 (3)--(4)，该点的 exact DFI contribution 满足

\[
\frac{\Delta(x)^2}{P(x)}
\ge
\frac{\tau}{2r^2}.
\tag{5}
\]

对所有 active 点求和得到完全确定性的引理：

\[
\boxed{
I_r(n)
\ge
\frac{\tau}{2r^2}|A_\tau(n,r)|.
}
\tag{6}
\]

这个下界没有 overlap approximation、没有 cutoff，也没有相减两个接近的大数。
若 \(P(x)=0<Q(x)\)，exact DFI 已为 \(+\infty\)，式 (6) 只需按
extended-value 意义理解。

## 3. 需要的统计规律

记 active fraction

\[
a_\tau(n,r)=\frac{|A_\tau(n,r)|}{N}.
\tag{7}
\]

与其分别猜测 active fraction 的 \(n\)-律和 \(D_r\) 的 \(r\)-律，更直接的量是

\[
G_\tau(n,r):=\frac{a_\tau(n,r)}{r^2}.
\tag{8}
\]

本轮采用下面明确标注的统计假设。

**Statistical active-count law.** 对 admissible 偶数非 dyadic 周期，

\[
\boxed{
G_\tau(n,r)
=\frac{|A_\tau(n,r)|}{Nr^2}
\ge c_Ae^{-\beta n},
\qquad \beta<\ln2.
}
\tag{H}
\]

数值支持一个更强的拟合；为了留出充分余量，最终建议使用

\[
c_A=1,\qquad \beta=0.55<\ln2\approx0.693147.
\tag{9}
\]

这里 \(r^2\) 不是事后添加的拟合因子。它来自式 (4) 中相邻 period 变化的自然
\(r^{-2}\) 尺度，并且会在式 (6) 中精确消掉。

## 4. 条件 DFI 定理

由 (6)--(8) 和假设 (H)，

\[
\begin{aligned}
I_r(n)
&\ge \frac{\tau}{2r^2}Na_\tau(n,r)\\
&=\frac{\tau}{2}N G_\tau(n,r)\\
&\ge\frac{\tau c_A}{2}
   e^{(\ln2-\beta)n}.
\end{aligned}
\tag{10}
\]

代入式 (2) 和 (9)，得到

\[
\boxed{
I_r(n)
\ge
1.5\times10^{-4}\,e^{0.14314718n}.
}
\tag{11}
\]

因此真正需要的条件是 \(\beta<\ln2\)，而不是旧 active-set 归约中的
\(\beta<\ln2/2\)。式 (10) 也不再需要另外假设 \(D_r\ge c_D/r^2\)。

## 5. 小规模全窗口 exact 检查

脚本

```text
scripts/experiments/hp1_active_tail_exact.py
```

使用 roots-of-unity point formula，但完整枚举全部 \(2^n\) 个输出。设置为

- exact support；
- \(2\le r<2^{n/4}\)；
- 偶数非 dyadic \(r\)；
- \(\tau=3\times10^{-4}\)。

对每个 \(n\)，在完整 period window 中最小化式 (6)，得到：

| \(n\) | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minimizer \(r\) | 6 | 6 | 6 | 10 | 12 | 14 | 18 | 22 | 26 | 30 |
| \(\log[\tau N a_\tau/(2r^2)]\) | -5.490 | -5.015 | -4.538 | -4.614 | -4.390 | -4.226 | -4.039 | -3.859 | -3.601 | -3.459 |

在 \(n=14,\ldots,20\) 上，归一化 active envelope 的拟合为

\[
\boxed{
\min_r\log G_\tau(n,r)
\approx -0.499916n+1.49303,
\qquad R^2=0.999554.
}
\tag{12}
\]

所以完整可枚举窗口支持 \(\beta\approx0.50\)。保守值 \(\beta=0.55\) 覆盖了
所有这些 exact 数据。

## 6. 为什么普通 Monte Carlo 不能上 200 比特

在 uniform output sampling 下，\(A_\tau\) 是 rare event。以 \(r=12\) 为例，
其 fraction 从 \(n=20\) 的约 \(7\times10^{-2}\) 下降到 \(n=200\) 的约
\(10^{-30}\)。因此样本数即使按 \(n^2\) 增长也不可能直接命中 active 点。
这正是此前 uniform CUDA 在大 \(n\) 上 ESS 崩溃的原因。

本轮新增

```text
scripts/experiments/hp1_active_tail_subset_cuda.cu
```

使用 adaptive subset simulation：

1. 从 uniform bitstrings 开始；
2. 每层保留 active score 的上分位数；
3. 重采样后使用对称 bit-flip Markov chain；
4. 各层 conditional fractions 相乘，估计 exponentially small count；
5. point probability 使用 log-domain roots-of-unity filter，避免 \(n\gtrsim100\)
   时中间 product 下溢。

这个方法不构造 statevector，支持 \(n\le300\)。对 bounded odd parts，每次 point
query 的代价是

\[
O\!\left(n\,[u_r+u_{r+1}]\right),
\tag{13}
\]

其中 \(u_s\) 是 \(s\) 的 odd part。

## 7. rare-event estimator 的小规模校准

直接可枚举或可靠 uniform MC 的结果为

\[
\begin{array}{c|c|c}
(n,r)&\text{reference }a_\tau&\text{subset simulation}\text{ 的典型结果}\\
\hline
(20,12)&0.06952095&0.069\text{--}0.072\\
(22,12)&0.03225803&0.0312\text{--}0.0331\\
(30,12)&0.00213275\pm2.31\times10^{-5}&0.00212\text{--}0.00226
\end{array}
\tag{14}
\]

所以 point-forward、event definition 和 subset-product normalization 在可核验窗口中一致。

## 8. \(n=200\) 结果

### 8.1 固定 \(r=12\) 的完整 scaling

使用 \(n=20,40,\ldots,200\)，每点 4 个独立 replicate（\(n=200\) 为 6 个），
得到

\[
\boxed{
\log a_\tau(n,12)
\approx-0.374866n+5.45864,
\qquad R^2=0.999753.
}
\tag{15}
\]

在 \(n=200\)，六个 replicate 给出

\[
\operatorname{mean}[\log a_\tau]=-69.5292,
\qquad
\operatorname{sd}[\log a_\tau]=0.6523.
\tag{16}
\]

### 8.2 不同 period 的归一化一致性

高粒子数复核使用 8192 particles、256 mutation steps。结果为

| \(r\) | replicates | mean \(\log a_\tau\) | mean \(\log(a_\tau/r^2)\) |
|---:|---:|---:|---:|
| 12 | 6 | -69.529 | -74.499 |
| 14 | 2 | -69.306 | -74.584 |
| 20 | 2 | -67.685 | -73.677 |
| 24 | 2 | -80.863 | -87.219 |

这些结果位于同一指数尺度，但也显示出不可忽略的 2-adic/period-class
差异：\(r=12,14,20\) 的归一化值接近，\(r=24\) 更低。因此不能把
\(a_\tau/r^2\) 当成 period-independent 常数；保守统计律 (17) 针对的是其
lower envelope。

保守假设

\[
\log G_\tau(n,r)\ge-0.55n
\tag{17}
\]

在全部 exact-envelope 点和上述 \(n=200\) 代表点上的最小 log margin 为

\[
\min_{\rm observed}\bigl[\log G_\tau+0.55n\bigr]
\approx1.741>0.
\tag{18}
\]

## 9. 必须保留的限制

1. **(H) 是统计规律，不是 uniform theorem。** 和 period-overlap Lemma 中的
   statistical law 一样，式 (H) 是经验闭合。
2. **200 比特没有枚举完整 period window。** 当 odd part 指数大时，当前 exact
   point query 本身不再是 polynomial。\(n=200\) 检查使用了代表性 bounded-odd-part
   periods；“对所有 admissible \(r\)”是明确写入 (H) 的假设。
3. **subset simulation 有 MCMC mixing error。** 因此保存独立 replicates、最低
   acceptance 和 distinct-state fraction。不能只保留一条回归线。
4. **本文只闭合偶数方向。** dyadic 偶数周期由 exact support mismatch 覆盖；
   非 dyadic 偶数周期使用 (H)。奇数 \(r\) 的 Pearson denominator 仍是 \(P_r\)，
   不能通过交换 \((r,r+1)\) 自动得到。
5. **exact DFI 与固定 cutoff DFI 不同。** 式 (5)--(11) 针对 exact DFI；固定
   \(\epsilon>0\) 的 regularized DFI 不可能渐近无界指数增长。

## 10. 可用于论文的条件表述

> For even non-dyadic periods, define the active small-denominator set by
> \(NP_r(x)<2\) and
> \((NP_{r+1}(x)-NP_r(x))^2r^2\ge\tau 2^n\). Each active output contributes at
> least \(\tau/(2r^2)\) to the exact Pearson DFI. Conditioned rare-event
> simulations, calibrated against exact enumeration and extended to 200 qubits,
> support the statistical law
> \(|A_\tau|/(2^nr^2)\gtrsim e^{-\beta n}\) with a conservative
> \(\beta=0.55<\ln2\). Under this explicitly empirical law, the DFI obeys
> \(I_r\gtrsim(\tau/2)e^{(\ln2-\beta)n}\).

原始与汇总数据见：

- `data/hp1_active_tail_exact/even_non_dyadic_n10_20.csv`
- `data/hp1_active_tail_subset_cuda/r12_n20_180_replicates.csv`
- `data/hp1_active_tail_subset_cuda/r12_n200_replicates.csv`
- `data/hp1_active_tail_subset_cuda/r14_n200_heavy.csv`
- `data/hp1_active_tail_subset_cuda/r20_n200_heavy.csv`
- `data/hp1_active_tail_subset_cuda/r24_n200_heavy.csv`
- `doc/ai/hp1_active_tail_n200_summary.json`
