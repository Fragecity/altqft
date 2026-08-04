#set page(
  paper: "a4",
  margin: (x: 2.1cm, y: 2.0cm),
)
#set text(
  font: "Noto Sans CJK SC",
  size: 10pt,
)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: none)
#set math.equation(numbering: "(1)")

= HP-1 DFI 的小分母机制：一个无额外假设的有限窗口证书

*工作笔记，2026-08-04。本文不修改 `appendix.tex`；所有数值均由当前仓库中的 HP-1 实现重新计算。*

== 结论先行

离散 Fisher information 的正增长并不是由碰撞量
$D_r / sqrt(S_(r,r))$ 解释的。这个量正是 E50 所能得到的 lower bound，数值上随 $n$ 衰减。真正的机制是逐点分母的加权调和平均迅速变小：变化最大的输出点恰好具有很小的 $P_r(x)$。

对任意两个有限概率分布，下面的分解是精确的，不需要 overlap ansatz、uniformity 假设或任何渐近假设：
$ I_r = underbrace(D_r / sqrt(S_r)) "collision bound" times underbrace(Gamma_r) "small-denominator amplification". $

数值上，前一个因子衰减，而后一个因子以指数趋势增长；两者相乘给出图中的正斜率。这是对小分母的直接解释。严格地说，有限的 $n=7..18$ 数值只能给出 finite-window certificate，不能单独给出 $n -> infinity$ 的 theorem。

== 1. 对象和数值约定

令 $N=2^n$，输出空间为 $Omega_N={0,1,dots,N-1}$。对周期 $r$ 和 shift $c=0$，使用真正的有限寄存器 support
$ S_r = {q r: q >= 0, q r < N}, quad R_r = abs(S_r), $
以及输入态的分量表示
$ psi_r(y) = cases(1 / sqrt(R_r), & y in S_r, \\ 0, & "otherwise"). $

给定 HP-1 unitary $U$，定义输出振幅和概率
$ A_r(x) = sum_(y in Omega_N) U_(x,y) psi_r(y), quad P_r(x)=abs(A_r(x))^2, $
$ Q_r(x)=P_(r+1)(x), quad Delta_r(x)=Q_r(x)-P_r(x). $

本文的 exact DFI（在没有零分母时）是
$ I_r = sum_(x in Omega_N) Delta_r(x)^2 / P_r(x). $

如果 $P_r(x)=0$ 而 $Q_r(x)>0$，采用 extended-value 定义 $I_r=+infinity$；若两者都为零，则该项定义为零。数值实现明确使用正的分母
$ tilde(P)_(r,epsilon)(x) = max(P_r(x), epsilon), quad epsilon=10^(-12), $
以及
$ I_(r,epsilon) = sum_x Delta_r(x)^2 / tilde(P)_(r,epsilon)(x). $
下面的分解对任意严格为正的分母向量都成立：exact 情形取 extended-value 极限，数值情形取 $tilde(P)_(r,epsilon)$。因此 cutoff 不是隐藏假设，而是被显式写入被核验的量；它必须被报告，因为极小概率正是本笔记研究的对象。

为避免把两个不同的 period window 混在一起，记
$ W_n^"num" = {2,...,n^2-1}, quad W_n^"th" = {r: 2 <= r < 2^(n/4)}. $
图 `fi_vs_nqubits.svg` 实际使用前者；E51 写的是后者。

== 2. 精确的小分母分解

令 $p_r(x)$ 表示实际用于分母的正向量：exact 正分母情形取 $p_r=P_r$，数值情形取 $p_r=tilde(P)_(r,epsilon)$。为简化记号，以下把它写成 $P_r$；相应地
$ D_r = sum_x Delta_r(x)^2, quad S_r = sum_x p_r(x)^2. $

=== 引理 1：有效分母的精确分解

若 $D_r>0$，定义变化权重
$ w_r(x) = Delta_r(x)^2 / D_r $
以及有效分母
$ P_"eff",r = (sum_x w_r(x) / P_r(x))^(-1). $

则
$ I_r = D_r / P_"eff",r. $

*证明。* 直接代入定义：
$ I_r = sum_x Delta_r(x)^2 / P_r(x)
      = D_r sum_x w_r(x)/P_r(x)
      = D_r/P_"eff",r. $

如果 $D_r=0$，则 $P_r=Q_r$，DFI 为零；此时结论以平凡方式成立。证明中没有使用任何关于 HP-1、period 或 $n$ 的额外假设。

=== 引理 2：E45 是这个分解的粗化

对每个 $x$，有 $P_r(x)^2 <= S_r$，故 $P_r(x) <= sqrt(S_r)$。于是
$ 1/P_r(x) >= 1/sqrt(S_r) $
并得到
$ I_r >= D_r/sqrt(S_r). $

定义
$ Gamma_r = sqrt(S_r)/P_"eff",r = I_r sqrt(S_r)/D_r. $

那么
$ I_r = (D_r/sqrt(S_r)) Gamma_r, quad Gamma_r >= 1. $

这说明 E45 并不是错误，而是把所有逐点分母都替换成全局上界后的结果。它完全丢掉了 $Gamma_r$。

=== 引理 3：单点和尾部证书

对任意非空输出集合 $A subset.eq Omega_N$，有精确 lower bound
$ I_r >= I_r(A) := sum_(x in A) Delta_r(x)^2/P_r(x). $

特别地，定义单点证书（式（20））
$ T_r := max_(x: P_r(x)>0) Delta_r(x)^2/P_r(x). $
则式（21）
$ I_r >= T_r $
只是由非负项求和得到的下界，不是 $T_r$ 的第二个定义。

如果 $A_(r,K)$ 是逐点贡献
$ Delta_r(x)^2/P_r(x) $
最大的 $K$ 个输出点组成的集合，则
$ I_r >= T_(r,K) := sum_(x in A_(r,K)) Delta_r(x)^2/P_r(x). $

这些都是非负项求和的直接结果，没有引入任何统计或渐近假设。这里的 $T_r$ 专指“最大单个 contribution”；$T_(r,K)$ 是把最大的 $K$ 个 contribution 相加后的更强证书。

== 3. 每一步的数值核验

数值脚本直接调用 `src/altqft/nn/process_qc.py` 的 state-vector evaluator。主表使用 exact support `arange(0,N,r)`；同时用旧的 surrogate support `N//r` 重复一次，以便和论文已有数据比较。$n=7..18$，$r in W_n^"num"$，共 $1994$ 个相邻 period pairs。

数值表格使用 cutoff 分母，因此严格说计算的是
$ T_(r,epsilon) := max_x Delta_r(x)^2 / tilde(P)_(r,epsilon)(x), quad I_(r,epsilon) >= T_(r,epsilon). $
它是式（20）的正分母、regularized 版本；当 $epsilon -> 0$ 且没有零分母时，它退化为 $T_r$。为简洁，后文表头中的 $T_r$ 均指这个数值版本，并在涉及 exact 定义时明确写出 $T_r$。

#table(
  columns: (4.2cm, 3.0cm, 5.8cm),
  table.header[*核验项目*, *最大误差/计数*, *核验内容*],
  [概率归一化], [$5.9 dot 10^(-7)$], [$abs(sum_x P_r(x)-1)$ 和 $abs(sum_x Q_r(x)-1)$],
  [平方距离恒等式], [$8.4 dot 10^(-17)$], [$D_r = S_(r+1,r+1)-2 S_(r,r+1)+S_r$],
  [有效分母分解], [$2.9 dot 10^(-16)$], [$I_r = D_r/P_"eff",r = (D_r/sqrt(S_r)) Gamma_r$],
  [碰撞 lower bound], [0 次违反], [$I_r >= D_r/sqrt(S_r)$],
  [单点 lower bound], [0 次违反], [$I_(r,epsilon) >= T_(r,epsilon)$],
)

归一化误差来自 circuit state 的 complex64 计算；概率代数本身在 double precision 上执行。平方距离、有效分母和单点证书的恒等式/不等式在全部 $1994$ 个 pairs 上通过。

下面的最小化和拟合只用于描述有限窗口，不被当作无穷维证明。

== 4. 数值上真正增长的因子

对每个 $n$，在 $W_n^"num"$ 中取实际 DFI 最小的 period $r_*(n)$，然后计算 $D_r$、碰撞 lower bound $B_r=D_r/sqrt(S_r)$、有效分母 $P_"eff",r$ 和放大因子 $Gamma_r$。exact-support 的三个代表点如下。

#table(
  columns: (0.9cm, 1.0cm, 1.5cm, 1.6cm, 1.6cm, 1.7cm, 1.5cm),
  table.header[*$n$*, *$r_*$*, *$I_r$*, *$D_r$*, *$B_r$*, *$P_"eff",r$*, *$Gamma_r$*],
  [10], [70], [10.386], [$2.206 dot 10^(-3)$], [0.04633], [$2.124 dot 10^(-4)$], [224.2],
  [14], [166], [33.630], [$2.683 dot 10^(-4)$], [0.01697], [$7.979 dot 10^(-6)$], [1982],
  [18], [90], [143.853], [$1.890 dot 10^(-4)$], [0.01161], [$1.314 dot 10^(-6)$], [12386],
)

对 $n=7..18$ 做 log-linear 描述性拟合，exact support 与 surrogate support 的结果为：

#table(
  columns: (4.0cm, 2.4cm, 2.4cm),
  table.header[*量*, *exact support 斜率*, *surrogate support 斜率*],
  [$log I_"min"$], [0.341], [0.368],
  [$log D_(r_*)$], [-0.457], [-0.487],
  [$log B_(r_*)$], [-0.228], [-0.225],
  [$log P_"eff",(r_*)$], [-0.798], [-0.856],
  [$log Gamma_(r_*)$], [0.569], [0.594],
)

因此数值关系是
$ 0.341 approx (-0.457)-(-0.798),
  quad 0.368 approx (-0.487)-(-0.856). $

这给出了小分母的说明：$D_r$ 本身衰减，但 $P_"eff",r$ 衰减得更快。DFI 是二者的比值，故可以增长。E50 只看到 $B_r$，没有看到 $P_"eff",r$ 或 $Gamma_r$。

在 exact-support 的 $n=18$ full-window minimizer 上，约 $97%$ 的逐点 DFI contribution 来自 $P_r(x)<10^(-6)$ 的 outcomes。这一事实与有效分母机制一致，也解释了为什么改变 cutoff 会改变拟合斜率。

== 5. 无假设的有限窗口 lower certificate

对数值 cutoff 版本定义
$ T_(min,epsilon)^(1)(n) = min_(r in W_n^"num") T_(r,epsilon). $

由引理 3 的 regularized 版本，对每个有限 $n$ 都有严格的数值可验证不等式
$ I_(min,epsilon)(n) = min_(r in W_n^"num") I_(r,epsilon)
  >= min_(r in W_n^"num") T_(r,epsilon)
  = T_(min,epsilon)^(1)(n). $

下面列出单点证书。每一个数都不是回归预测，而是对该有限窗口中所有 period 的逐点枚举后取 minimum。

#table(
  columns: (0.8cm, 2.0cm, 2.0cm),
  table.header[*$n$*, *$T_(min,epsilon)^(1)$ surrogate*, *$T_(min,epsilon)^(1)$ exact*],
  [7], [0.0618], [0.1046],
  [8], [0.2089], [0.2820],
  [9], [0.5707], [0.4205],
  [10], [0.9299], [1.0208],
  [11], [0.4277], [2.1077],
  [12], [1.0069], [1.0134],
  [13], [1.5480], [2.3472],
  [14], [2.6112], [2.4615],
  [15], [3.1026], [1.1787],
  [16], [5.9889], [4.5974],
  [17], [6.9945], [4.6392],
  [18], [9.5399], [5.8343],
)

在 $n=7..18$ 上，单点证书的 log-linear 拟合为：

- surrogate support: $k=0.403$, $R^2=0.927$, 95% CI 为 $[0.324,0.483]$；
- exact support: $k=0.308$, $R^2=0.822$, 95% CI 为 $[0.207,0.410]$。

这比直接拟合 $I_(min,epsilon)$ 更保守：它只使用一个 output outcome 的一个 nonnegative DFI term。因此，它是一个真正保留小分母的 finite-window numerical certificate，而不是把小分母平均掉的 collision bound。

在 exact、无 cutoff 的记号中，若 $x_*$ 是式（20）的 argmax，则
$ T_r = Delta_r(x_*)^2 / P_r(x_*). $
数值表对应的 regularized 版本则是
$ T_(r,epsilon) = Delta_r(x_*)^2 / tilde(P)_(r,epsilon)(x_*), $
其中 $x_*$ 最大化 regularized contribution。对 $T_(min,epsilon)^(1)$ 对应的 $(r_*,x_*)$ 序列，逐点数值拟合给出：

- surrogate support: $log P_r(x_*)$ 的 slope 为 $-1.473$，$log abs(Delta_r(x_*))$ 的 slope 为 $-0.535$，所以 $2(-0.535)-(-1.473)=0.403$；
- exact support: 对应 slopes 为 $-1.563$ 和 $-0.627$，所以 $2(-0.627)-(-1.563)=0.309$。

这不是新的假设，而是逐点恒等式
$ log T_(r,epsilon) = 2 log abs(Delta_r(x_*)) - log tilde(P)_(r,epsilon)(x_*) $
的数值分解。它直接显示了机制：分母概率的指数衰减速度超过了差分振幅平方的衰减速度。

但这不等于说 $I_r$ 的数值主要等于单个 $T_r$。$T_r$ 是一个严格的 lower certificate；$I_r-T_r$ 仍包含其余所有 rare-outcome terms。比如在 $n=18$ 的 $I_r$ minimizer 上，exact support 给出 $I_r approx 143.9$、最大单项约 $10.66$；surrogate support 给出 $I_r approx 128.4$、最大单项约 $9.54$。因此更准确的表述是：$T$ 的有限窗口指数增长已经足以证明并展示增长机制，但 $I$ 的完整幅度来自许多小分母项的总和。

也可以取最大的 $K$ 个逐点 terms 求和。该量仍然严格小于等于 DFI；它只提高证书的数值大小，不改变证明逻辑。

== 6. E50/E51 的准确关系

E50 的推导在 overlap law 成立时给出一个 collision-level lower bound。数值重新计算表明，这个 bound 的斜率约为 $-0.23$，与 $1/r$ 型衰减一致。因此 E50 和 overlap 数值并不矛盾。

真正不能成立的是把
$ I_r >= D_r/sqrt(S_r) = Omega(1/r) $
解释成 DFI 的 positive exponential growth。这个 lower bound 太弱；它允许真实 DFI 由 $Gamma_r$ 提供额外的指数因子。

此外，当前数值图的窗口是 $W_n^"num"$，不是 E51 写的 $W_n^"th"$。直接在 $W_n^"th"$ 上取 minimum 的数值会受到极小概率和 extended-value/cutoff 规则影响，并没有复现图中平滑的正斜率。例如当前 epsilon=10^(-12) 的直接扫描给出：

#table(
  columns: (0.8cm, 2.6cm, 2.6cm),
  table.header[*$n$*, *$min_(r in W_n^"th") I_r$ surrogate*, *$min_(r in W_n^"th") I_r$ exact*],
  [10], [12140], [4077],
  [11], [284], [10500],
  [12], [913], [656],
  [13], [1885], [6807],
  [14], [1028], [1344],
  [15], [1608], [1924],
  [16], [3208], [9359],
  [17], [3568], [3141],
  [18], [6006], [7536],
)

因此，本笔记的证书严格支持的是“$n=7..18$、$r<n^2$ 的 finite-window empirical lower envelope”。它不能在不增加新证明的情况下把 E51 的 theorem window 推广成 asymptotic theorem。

如果要证明 E51 的原始 theorem window，仍需证明一个结构性结论，例如对每个 $r in W_n^"th"$ 构造 output 集合 $A_(r,n)$ 并直接控制
$ P_r(x) $ 的小分母和
$ sum_(x in A_(r,n)) Delta_r(x)^2/P_r(x) $。
这个控制不能由三个 collision overlaps 单独推出。

== 7. 可复现实验骨架

以下代码骨架展示每个 certificate 的计算方式；`exact_support=True` 对应本笔记的主表，改为 `False` 即得到论文旧数据使用的 surrogate support。

```python
import torch
from altqft.circuits.HPgenerators import HP1
from altqft.nn.process_qc import _torch_circuit_probability_vectors

device = torch.device("cuda")
eps = 1e-12
for n in range(7, 19):
    U = HP1(n)
    values = []
    for r in range(2, n*n):
        P, Q = _torch_circuit_probability_vectors(
            U, [r, r + 1], 0, exact_support=True, device=device
        ).double()
        delta2 = (Q - P).square()
        terms = delta2 / P.clamp_min(eps)
        I = terms.sum()
        D = delta2.sum()
        P_pos = P.clamp_min(eps)
        Q_pos = Q.clamp_min(eps)
        S_raw = P.square().sum()
        S_next_raw = Q.square().sum()
        X_raw = (P * Q).sum()
        S = P_pos.square().sum()
        P_eff = D / I
        Gamma = I * S.sqrt() / D
        T = terms.max()
        assert abs(float(P.sum() - 1)) < 1e-5
        assert abs(float(Q.sum() - 1)) < 1e-5
        assert abs(float(D - (S_next_raw - 2*X_raw + S_raw))) < 1e-8
        assert abs(float(I - D / P_eff)) < 1e-8
        assert abs(float(I - (D / S.sqrt()) * Gamma)) < 1e-8
        assert float(I + 1e-9) >= float(D / S.sqrt())
        assert float(I + 1e-9) >= float(T)
        values.append((float(I), float(T), float(D), float(P_eff), float(Gamma)))
    I_min = min(row[0] for row in values)
    T_min = min(row[1] for row in values)
```

所有恒等式的数值核验都应在保存数据前执行；不能只保留 log-linear fit 而不保留逐点 $P_r,Q_r,D_r,P_"eff",r$。

== 8. 最终表述建议

可以安全地写成：

#block(
  stroke: (left: 1.5pt + rgb("3b82f6")),
  inset: 8pt,
)[
  The observed finite-window growth of the HP-1 DFI is explained by a small-denominator mechanism.  The collision-based lower bound captures only the squared $L^2$ separation of adjacent output laws.  An exact decomposition retains the effective, change-weighted harmonic denominator $P_"eff",r$; numerical enumeration shows that this denominator decreases faster than the squared-distance factor over the tested window.  A single-output contribution already gives a denominator-aware finite-window lower certificate.  This is an empirical finite-size result, not by itself an asymptotic theorem.
]

这段说明没有把数值拟合伪装成证明，也没有引入未经验证的 overlap/uniformity 假设；同时它明确指出了为什么 DFI 可以指数增长，以及 E50 为什么不能捕获该增长。
