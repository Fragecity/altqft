# 《An alternative to the Quantum Fourier Transform》对抗性审查报告

> 审查对象：`main.tex`、`referee_reply/referee.tex`、`referee_reply/reply.tex`、`appendix.tex`  
> 审查性质：数学、算法、复杂度、数值实验、审稿回复覆盖度、修订映射及制作质量的独立对抗性审查  
> 行号说明：文中 `main:53` 等均指压缩包内当前版本的源文件行号。

## 0. 结论先行

**当前版本不宜直接提交，也不足以逆转 PRL 的拒稿决定。** 这不是因为项目没有结果，而是因为稿件中最强的四个中心主张——“可替代 QFT”“一般 HSP”“指数增长定理”“高效神经网络后处理”——明显超过了当前数学和实验实际支持的范围。

可以严格或较稳妥保留的核心是：

1. 对 shift-invariant unitary 的平坦模矩阵元必要条件（`main:98–113`；`appendix:165–230`）；
2. 对 **真正的** \(\mathbb Z_{2^n}\) 子群及其陪集，HP 双线性相位电路的精确 shift invariance（`main:140–147`；`appendix:232–325`）；
3. HP-0 在 dyadic 子群上的解析恢复示例与失败概率 \(2^{-m}\)（`appendix:329–439`）；
4. \(n=7,\ldots,18\) 的有限窗口 DFI 描述性趋势、固定 \(n=18\) 的 510 类 Deep Sets 分类器，以及特定噪声模型下的有限规模 stress test（`appendix:1164–1377`）。

当前不能成立或不能由现有证据支持的主张包括：

- HP-1 已构成一般 Shor/QFT replacement；
- Theorem 2 证明了正指数增长的 DFI；
- DFI 推出了可达的高效样本复杂度；
- 当前 510-logit 分类器构成随 \(n\) 多项式扩展的 decoder；
- 18-bit 条件化 benchmark 已证明 Shor scalability；
- HP 是 \(O(n)\) depth 而标准 QFT 是 \(O(n^2)\) depth 的同口径渐近优势。

建议二选一：

- **窄化路线（推荐）**：把论文改成“dyadic HSP 上的精确 shift-invariant HP 电路 + 截断周期态上的有限规模 learned decoding”。撤销或降格 Theorem 2，删除 replacement / efficient / general HSP / verified scalability 等措辞。
- **强主张路线**：保留现题目时，必须补严格的全域分离与可达样本界、端到端多项式 decoder、同一资源模型下的 QFT 基线，以及随机 base、完整寄存器与失败事件均计入的 Shor 实验。

## 

## 3. 六组对抗审查的裁决摘要

### 3.2 第 2 点：审稿人真正是什么意思

**支持方最强论点。** Referee A 说的是中心复合主张“尚未建立”，不是给出 HP-1 不可能工作的反例；Referee B 基本认可工作，只要求形式、叙事和噪声展示改进。

**反对方最强论点。** Editor 的信是明确终局拒稿且不建议 APS 转投，不是 ordinary major revision。A1–A5 是一条连续否定链：toy 结果不能支撑 Shor → invariance 不是 recovery → DFI 下界不能给可达上界 → 小规模不能给 scalability → NN 掩盖了缺失的 decoding theory。

**裁决。** 两者并不矛盾：A 没有证明 HP 思路错误，但已明确认定当前证据不能支持标题/摘要中的组合主张；若申诉，举证负担是决定性新结果或彻底缩小 claim，而不是逐条润色。B 的正面意见不能抵消 A 对 validity/complexity 的核心否定。

### 3.3 第 3 点：reply 有没有漏答

**支持方最强论点。** A3、A4、A6、B1、B2 和 B3 的作图要求都有直接回复；A1 的特殊性、A2 的 distinguishability 也可在其他段落中找到间接回应，不能因没有逐字重引就判为“没答”。

**反对方最强论点。** reply 有选择性截断：A1 漏掉 classical comparison；A2 漏掉“distinguish general periods and admit efficient recovery”；A5 漏掉 “small-n polynomial-size architecture \(\neq\) polynomial-cost Shor decoder”及 efficient claim unsupported；Editor-level appeal 也没有建立。

**裁决。** 多数编号问题有文字覆盖，但存在三个确定的实质遗漏和一个战略遗漏：A1 的经典基线、A5 的端到端 polynomial-cost 问题、B3 的 \(\eta\) 取值理由，以及若向原编辑申诉时缺少明确 reconsideration 请求与 PRL 六项门槛论证。A2 属于“分散谈到但没有闭环”，不宜简单归为完全漏答。

### 3.4 第 4 点：哪些回复得不好

**支持方最强论点。** reply 最有价值的部分是主动承认边界：HP-0 是 toy、shift invariance 仅必要、有限数值不证明渐近、NN 没有 continued-fraction 类解析理论、DFI 对 NN 不是可达 bound。这些让步能支撑一篇更窄、更诚实的论文。

**反对方最强论点。** reply 每次让步后又恢复同一强主张：承认无 upper bound 后又说 efficient sample complexity；承认不 generalize 后又称 general HSP framework；承认 finite-size 后又说 demonstrate scalability；称删掉 HP-0 但正文和附录仍保留。

**裁决。** 回复质量的核心失败是**内部自我抵消**。诚实让步值得保留，但必须同步修改标题、摘要、正文和总结。Theorem 2 不能靠措辞继续当 theorem；18-qubit、NN 与 Shor 只有在降格为 finite/conditioned benchmark 后才可辩护。

### 3.5 第 5 点：正文与附录还有什么缺陷

**支持方最强论点。** Theorem 1 和 HP-0 的窄域数学成立；DFI、Deep Sets 与 noise sweep 可作为有限规模 diagnostic/engineering benchmark；许多缺陷通过精确定义和降 claim 可以修复。

**反对方最强论点。** 定义索引、对象域、DFI 零分母、QFT comparator、Theorem 2、指数 decoder、base 后选择和资源口径构成多个独立技术断点，并非单纯表达问题。

**裁决。** 可保留的数学核心没有被这些问题抹掉，但现稿的中心算法结论确实没有建立。第 6–7 节给出按 P0/P1/P2 分级的完整缺陷表。

### 3.6 第 6 点：实际修订是否都呈现在 reply

**支持方最强论点。** reply 已覆盖最重要的可见修订：HP-0 降权、HSP/shift 动机、删除旧 Eq. (8)、DFI/MLE 叙事、Theorem 2、18-qubit、decoder 限制、noise 曲线、notation 和组织。

**反对方最强论点。** 新增 QFT alternatives、HP-0 新采样保证、n=200 empirical closure、实际 \(r\le n^2\) DFI 窗、全 shift 缓存/验证划分、classifier-not-beam、精确网络和训练成本等没有呈现或被错误呈现。

**裁决。** 主线修改大多被“提到”，但几项决定结论强弱的事实没有透明披露，尤其是经验 closure 被称为 analytical result、实际 period window 被写错、固定分类头被称 beam/poly decoder、HP-0 被称 removed。第 8 节给出逐项映射。

## 4. Referee 与 Editor 的逐项含义和覆盖矩阵

| 来源                    | 真正要求/含义                                                             | reply 覆盖                                             | 当前判定                                                                  |
| --------------------- | ------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| Editor `referee:8–12` | 明确 reject；六项 PRL 门槛未形成 compelling case；无 APS 转投                     | `reply:83–86` 仅致谢和列变化                                | 战略性重大缺口；若申诉需明确请求及逐项 case                                              |
| A 总评 `referee:34`     | 复合主张“\(O(n)\) HP-1 + efficient postprocessing 替换 \(O(n^2)\) QFT”未建立 | `reply:228–233` 说 scope 更精确                          | 部分；顶层 claim 实际未改                                                      |
| A1 `referee:38`       | HP-0 仅 dyadic toy；经典查询即可解；无量子算法进步；不支撑 Shor                          | `reply:241–251` 承认 toy 并移附录                          | 大部覆盖；遗漏 \(f(0)\) vs \(f(2^j)\) 经典基线及“不主张量子优势”                         |
| A2 `referee:40`       | shift invariance 仅必要；要证明一般 periods 可区分并高效恢复                         | `reply:261–273` 承认 necessary/not sufficient；以 DFI 回应 | distinguishability 有有限证据；efficient recovery 未闭环                       |
| A3 `referee:42`       | HCR 是样本数下界；小下界不证明存在高效 estimator；需可达上界/具体 decoder                    | `reply:282–347` 删除 Eq. (8) 并承认无 upper bound          | 形式完整；`reply:347,673`、`main:287` 又重犯，实质失败                              |
| A4 `referee:44`       | 小规模数值与有限 \(r\) 窗不证明 scalability/general HSP                         | `reply:365–416` 扩至 18q 并承认 finite                    | 有回应；但 \(2^{n/2}\) 被误称 polynomial，且 general HSP 表述冲突                   |
| A5 `referee:46`       | NN 填补缺失理论；小 \(n\) poly-size 网络不等于 poly-cost Shor decoder            | `reply:438–522` 承认无解析 decoder、补 MLE                  | 核心复杂度句被漏引且未回答；重大遗漏                                                    |
| A6 `referee:48`       | PRL 正文过技术                                                           | `reply:560–563` 移细节、加动机                              | 基本覆盖；实际可读性仍需编辑判断                                                      |
| B1 `referee:56`       | 首次出现时定义 \(\mathbb Z_{2^n}\) 和 Shor 的 \(r\)                          | `reply:632–639`；`main:140,376`                       | 内容定义正确；但 \(\mathbb Z_{2^n}\) 已在 `main:63` 先出现，严格说未做到 first appearance |
| B2 `referee:58`       | 解释原 Eq. (8) 的物理/统计动机                                                | `reply:645–675` 删除 Eq. (8)，改为 DFI/MLE 语境             | 形式覆盖；efficient 残句造成叙事矛盾                                               |
| B3 `referee:60`       | 严格解释 \(\eta\) 取值；表改 accuracy-vs-\(\eta\) 图                          | `reply:680–713` 定义 channel、20 点曲线                    | 图已完成；为何选端点/几何间隔仍未回答                                                   |

## 5. reply 的遗漏清单与最小补写

### 5.1 确定遗漏

1. **A1 classical baseline。** `reply:238` 在 “much simpler” 后截断了 Referee A 的具体论据。应明确写：只有 \(n+1\) 个候选 dyadic 子群，可通过比较 \(f(0)\) 与 \(f(2^j)\) 经典识别；作者不对 HP-0 主张量子优势，其作用仅为教学性展示 shift-invariance mechanism。
2. **A5 polynomial-cost accounting。** `reply:419–522` 没有直接回答输出类别数、参数数、训练分布生成、缓存、推理 shots、候选验证随 \(n\) 如何增长。必须二选一：给端到端 polynomial bound，或撤下题目/摘要中的 efficient claim。
3. **B3 \(\eta\) 选择理由。** 定义 channel 不是选择区间的理由。若只是探索性 stress test，应直说区间用于跨数量级定位退化区、并非硬件校准；若称 hardware-relevant，必须给目标平台错误率映射和置信区间。

### 5.2 谈到但没有闭环

- **A2 efficient recovery。** 建议补一个分层结论段：已证的是 dyadic shift invariance；有限证据是 DFI/18q classifier；未证的是一般 period 的 polynomial-time recovery。
- **Editor appeal。** 若这是给原 PRL Editor 的申诉，应写明 “we respectfully request reconsideration”，再按 validity、novelty、importance/broad interest 分列决定性新证据和撤回的主张。若不是申诉，不应把它写成 ordinary revision response。

## 6. 阻止投稿级技术问题（P0）

### 6.1 精确定理与 Shor 数值对象不一致

Theorem 1 的证明依赖 \(V=\langle2^s\rangle\le\mathbb Z_{2^n}\)，其陪集表现为固定低 \(s\) 位，因此 shift 只贡献可提出的纯相位（`appendix:295–324`）。而 DFI/Shor 使用

\[
|\psi_r\rangle=R^{-1/2}\sum_{q=0}^{R-1}|qr\rangle,
\qquad
R=\left\lfloor\frac{2^n-1}{r}\right\rfloor+1
\]

（`appendix:443–455`）。当 \(r\nmid2^n\) 时，这不是 \(\mathbb Z_{2^n}\) 的子群态，shift 后还可能改变项数。

独立精确检查采用附录 fixed-phase 矩阵（`appendix:675–697`）。在 \(n=3,r=3\) 时：

| shift   | 截断 support    | 输出概率                                        |
| ------- | ------------- | ------------------------------------------- |
| \(c=0\) | \(\{0,3,6\}\) | \((3/8,1/24,1/24,1/24,1/24,3/8,1/24,1/24)\) |
| \(c=2\) | \(\{2,5\}\)   | \((1/4,1/8,0,1/8,1/8,0,1/8,1/4)\)           |

两者总变差距离为 \(13/24\)，不是零。因此正文必须把“exact dyadic theorem”和“generic truncated-period empirical study”分开；若要用于 Shor，需要证明/测量对所有 \(c\) 的近似 shift invariance 及其随 \(n,r\) 的误差界。

### 6.2 Theorem 2 目前不是定理，且增长方向写反

所谓 HP-1 period-law overlap 在定理陈述中使用 “\(\approx\)”（`appendix:649–667`）；证明明确说核心 mixed term 来自 \(n=200\) 的 conditioned Monte Carlo，并称为 “empirical closure”（`appendix:820–827,919–954`）。将经验拟合代入不等式，不能得到无条件 analytical theorem。

后续证明还只处理

\[
\{\nu_2(r),\nu_2(r+1)\}=\{0,1\},\qquad 2\kappa_1>\kappa_0
\]

（`appendix:1097–1136`），却没有把这些条件写进 `main:275–285` 或 `appendix:957–967` 的定理，也没有覆盖 \(r=3\to4\)、\(7\to8\) 等相邻类。

最致命的是，证明实际只得到

\[
\mathrm{DFI}(r,n)=\Omega(1/r)
\]

（`appendix:1133–1143`）。在 \(r<2^{n/4}\) 的最坏端，这至多给出 \(\Omega(2^{-n/4})\) 的**衰减下界**；虽然可形式化写成 \(\exp(kn+b)\) 且 \(k<0\)，但绝不能称 positive exponential growth，更不能推出 efficient sample complexity（`main:273–287`；`reply:333–347,659–675`）。

可选修复：

- 降为 “conditional empirical overlap ansatz”，只报告受限 2-adic 类的 \(\Omega(1/r)\) 机制猜想；或
- 补带误差界的严格 overlap 定理、所有相邻类和 \(k>0\) 的证明，并重算图表。

### 6.3 DFI 定义、零分母及 QFT 比较不一致

`main:190–194` 的量实际是相邻分布的 Pearson divergence

\[
\chi^2(P_{r+1}\Vert P_r)=\sum_x\frac{(P_{r+1}(x)-P_r(x))^2}{P_r(x)},
\]

它非对称，而且没有规定 \(P_r(x)=0\) 时如何处理。附录证明只说 “each nonzero denominator”（`appendix:1024–1039`），相当于绕开了定义问题。

独立检查在同一 fixed HP-1、\(n=3\) 下得到

\[
P_2=(1/2,1/4,0,1/4,0,0,0,0),
\]

而上面的 \(P_3\) 在多个 \(P_2=0\) 的位置严格为正，因此 \(\chi^2(P_3\Vert P_2)=\infty\)。从 \(n=3\) 到 \(n=7\) 的精确扫描中，大量相邻对均出现这一现象。若数值图给出有限值，必须披露究竟是 smoothing、threshold、删项还是 extended-value 约定；不同约定会改变 scaling。

QFT 附录计算的却是把整数 \(r\) 连续化后的 derivative FI（`appendix:503–548`），不是同一个 finite-difference Pearson divergence；求导时把 \(R\) 当常数，之后又用 \(R\approx2^n/r\)（`appendix:638–644`）。Lemma 写 \(1\ll r\ll2^n\)，证明实际要求 \(1\ll r\ll R\)，即约 \(r^2\ll2^n\)。所以当前 HP/QFT exponent 比较不是同口径基线。

修复应让两者使用同一个离散 divergence、同一输入态、同一 period window 和明确的零质量规则；若保留连续 QFT FI，只能称 heuristic comparator。

### 6.4 decoder 不是随 \(n\) 多项式扩展的已证算法

固定 \(n=18\) 的 Deep Sets 主干适合无序样本，且附录对结构披露较好；问题在输出与数据生成：

- 一候选 period 一个 logit；\(n=18\) 为 510 类和 1,909,230 参数（`appendix:1305–1309`）；
- 若候选域真为 \(r\le2^{n/2}\)，输出层宽度即 \(\Theta(2^{n/2})\)；
- 对每个 \(r\) 的每个 shift 缓存分布，row 数为 \(\sum_{r\le2^{n/2}}r=\Theta(2^n)\)（`appendix:1322–1329`）；完整分布又含 \(2^n\) 个概率（`appendix:1265–1266`）；
- 所有 510 个 period label 都出现在训练，只 hold out 每个 period 的一个 shift；这不是 unseen-period 或 unseen-\(n\) 泛化；
- checkpoint 用同一 validation top-1 选择，仍无独立 test set（`appendix:1337–1343`）；
- “beam search” 实际只是排序固定的 510 logits（`appendix:1311–1316`），与 `main:369–373` 及图注不符。

因此能成立的表述是 “fixed \(n=18\), 510-class finite-regime classifier”。要保留 efficient decoder，需要结构化输出、端到端训练/推理/验证复杂度上界，以及 unseen periods 与 unseen \(n\) 的外推实验。

### 6.5 18-bit benchmark 不是一般 Shor scalability 证明

标准 order-finding/factoring 的资源背景可参见 [Shor 的原始论文](https://arxiv.org/abs/quant-ph/9508027)；当前图表没有达到该算法的端到端资源核算标准。

`main:376` 先“选择”满足未知 order \(r\le2^{n/2}\) 的 base \(a\)。但求 \(r\) 正是 order-finding 的核心；除非将其明确称为**按小 order 条件化的 benchmark**，否则这是依赖答案的筛选。正文没有给随机/预先规定 base 的成功概率、尝试次数、奇数 order、\(a^{r/2}\equiv-1\)、GCD 验证、oracle 调用或总 shot 成本。

18-bit \(N\) 只用 \(n=18\) 的 period register，也不能直接与标准 continued-fraction Shor 的精度资源比较；这里之所以可用，是因为先将 label 限到 2–511。附录没有 Shor/base/factoring protocol 章节，因而图 `main:351–380` 无法从四个正式文件独立复现。

图例报告 \(k=1\) 有 3590 个、\(k=2\) 有 1051 个。独立计数表明 \(225001\ldots254999\) 有 6405 个 semiprimes（含 5 个素数平方；若只算不同素数乘积则 6400）。**若图确实包含区间内全部 semiprimes**，则可推断 solved 为 \(4641/6405\approx72.46\%\)，grey/out-of-range 为 \(1764/6405\approx27.54\%\)。该比例是基于图例和区间声明的推断，稿件应直接报告正式 denominator、纳入规则与失败分解，而不是让读者反推。

### 6.6 \(O(n)\) vs \(O(n^2)\) 的资源比较口径错误

formal balanced HP-1 cross-partition block 有

\[
|\Lambda_1||\Lambda_2|=\Theta(n^2)
\]

个 CP 门（`main:124–135`）。在 all-to-all、互不共享量子位的两比特门可并行的模型下，可用 matching/edge coloring 排成 \(\Theta(n)\) 层；这支持 HP 的 linear two-qubit depth。但标准 exact QFT 在**同一并行模型**下同样有 \(O(n)\) depth，而不是 `main:53,59,376` 所称的 \(O(n^2)\) depth。若比较 gate count，则两者都是 \(\Theta(n^2)\)。稿件自己还引用了更浅的并行 approximate QFT（`main:62–63`）。

因此标题的复杂度对比是 apples-to-oranges。应给相同 connectivity、native gate set、rotation synthesis precision、ancilla、routing、success criterion 下的 gate count、two-qubit depth、shots 和 classical cost 表。`appendix:1371` 的 18q noise circuit 只有 51 个 CP，而 balanced 9×9 formal block有 81 个候选 CP；如有剪枝、零相位或稀疏化，应明确规则及 scaling。

[Cleve–Watrous 的原始结果](https://arxiv.org/abs/quant-ph/0006004)已经给出 approximate QFT 的 \(O(\log n+\log\log(1/\epsilon))\) depth，并说明此前基线深度为 \(O(n)\)，进一步表明正文把“门数”写成“层数”的比较不可维持。

### 6.7 HP-L 命名存在 off-by-one

正文说 HP-\(L\) 有 \(L\) 个 phase blocks、HP-1 有一个、HP-0 是单层 H（`main:75,171–182`）。但正式定义

\[
U_{\rm HP}^{(L)}=H_{\Lambda_L}\prod_{l=L-1}^{1}(CP_l H_{\Lambda_l})
\]

（`main:125–135`）只有 \(L-1\) 个 CP blocks；按此式 HP-1 没有 CP。实验所谓 HP-1 又使用 \(\Lambda_1,\Lambda_2\) 和一个 CP block（`main:341`；`appendix:675–697`）。代数思路可修，但当前 theorem 对象与实验标签不一致。建议用 phase-block 数 \(B\) 作为索引，定义 \(B+1\) 个 H partitions。

## 7. 高优先级与制作级缺陷

### 7.1 period windows 三套冲突

- Theorem 2：\(r<2^{n/4}\)（`main:266–284`；`appendix:957–967`）；
- 主文 DFI 数值描述：\(r\le2^{n/2}\)（`main:336–344`）；
- 附录称 Fig. 实际扫描：\(r\le n^2\)（`appendix:1164–1168`）；
- decoder：固定 \(r=2,\ldots,511\)（`appendix:1305–1309`）。

不同任务可以用不同窗口，但必须明确区分。`reply:379` 把 \(2^{n/2}\) 称 “polynomial period window” 是数学错误。先核对生成 Fig. 的代码；若实际是 \(n^2\)，统一文字；若图/回归混用窗口，必须重算。

### 7.2 数值和训练复现不足

- optimized HP-1 的 phase optimization 方法、objective 实现、seed、最终 phases 与为什么只有 51 个 CP 未完整给出；
- random HP-L 曲线的随机实例数和生成规则不充分；
- MLE 图的候选集、trial 数、shift、失败定义和 CI 不充分；
- 按附录训练设置估算，8192 outcomes/item × 1024 items/epoch × 1919 epochs 约为 161 亿次 synthetic draws；这不等于量子 shots，但应披露训练预算；
- inference 每实例使用 \(331{,}776\) shots（`appendix:1317–1320`），需要与 QFT/其他 baseline 做任务级成本比较。

### 7.3 noise 结论过强

noise sweep 是比旧 global mixture 更好的 stress test，但仍只用 shift \(s=0\)、每 period/\(\eta\) 四条 noisy trajectories、无误差条、多 seed 或 QFT baseline（`appendix:1369–1377`）。图的前几个低噪点有回升，故 `main:384` 和 `appendix:1377` 的 “monotonic” 不准确，应改 “overall decreasing”。横轴 “noisy strength” 应改 “per-gate Pauli error probability \(\eta\)”。

### 7.4 正文公式和 HSP 表述

- `main:82` 的 \(|V\rangle\) 第一项未归一化，\(q\) 被写成无限集合，且 \(V=\{|qr\rangle\}\) 混淆群元素与 ket；
- 一般 finite Abelian HSP 的隐藏子群不必由单一 generator \(r\) 生成（`main:79`）；
- `main:336` 的 probability formula 缺少 \(1/R\) normalization，附录 `711–725` 才是正确形式，也未在主文写 shift；
- `main:338` 的 `DFI_min(n): min` 缺等号；
- section title `main:171` 写 HSP over \(\mathbb Z\)，内容却混用 \(\mathbb Z_{2^n}\) 和 \(\mathbb Z_q\)；
- `appendix:436` 称 sample complexity “logarithmic in the number of qubits”，但 `439` 的结果是 \(\lceil\log_2(1/\epsilon)\rceil\)，对 \(n\) 独立，应改为 logarithmic in inverse failure probability；
- `appendix:101–119` 与 `121–157` 重复推导 Fourier sampling；`107` 还有 \(h/v\) 变量笔误；
- `appendix:234` 说 shift theorem “guarantees reconstruction”，与正文承认 necessary-not-sufficient 冲突。

### 7.5 图表叙事

- DFI 图 caption `main:181–182` 称 “Average DFI”，而正文/附录实际描述为每个 circuit 先对 \(r\) 取 minimum，再对随机 circuit 取 mean；应写清聚合顺序；
- training 图与 `main:378` 的 “within a few epochs” 不符：checkpoint 在 epoch 1669，训练到 1919（`appendix:1340–1343`）；
- Shor 像素图没有直接给总数、success rate、grey count、base inclusion rule，视觉上也难解释每行含义；
- noise 图无 CI，并且文字声称严格单调；
- `main:386` 的 “verify scalability” 远强于有限 \(n=7–18\) 描述性拟合。

### 7.6 引用的适用范围

[Kwon–Lie–Jiang](https://arxiv.org/abs/2602.21510)讨论的是一般参数框架中、MLE 和小误差条件下由 inverse Fisher matrix 控制的上下样本界；它不自动覆盖当前离散 label、相邻 Pearson divergence 和神经分类器。 [Weimar 等](https://arxiv.org/abs/2509.02407)研究连续参数估计中 Fisher information 如何流经 ANN，也不是当前 decoder 获得 polynomial sample/compute complexity 的证据。 [Deep Sets](https://arxiv.org/abs/1703.06114)支持 permutation-invariant architecture 的选择，但不证明 period decoder 的复杂度或跨 \(n\) 泛化。reply 和正文应明确这些引用只支撑局部方法动机。

### 7.7 reply 的内部矛盾与表述

- `reply:283,309,320` 承认 DFI 不给 upper bound；`347,673` 又说 suggests efficient sample complexity；`675` 再否认 upper bound；
- `reply:392–395` 否认 arbitrary HSP generalization；`409–416` 随即称 general HSP framework；
- `reply:377` 承认 finite experiments 不证明 asymptotic；`381` 又说 demonstrate scalability；
- `reply:251` 准确说只移除 HP-0 独立 section；`563` 却说 “removed the HP-0”；
- reply 仍以原强题目开头（`77`），而 `main:37,53,75–77,386` 也未真正缩小 claim。

### 7.8 LaTeX 与制作

`reply.tex` 四张图混用两种相对路径：

- `../fig/...`：`reply:370,699`；
- `fig/...`：`reply:429,479`。

从项目根编译会先找不到前一种，从 `referee_reply/` 编译会找不到后一种，因此不存在无需改路径即可完整编译的 cwd。回复还重复定义三个 active labels（Theorem 2 和两条方程在 `reply:331–342` 与 `657–668`），日期仍为 25 June 2026（`reply:79`），并留有多处中文工作注释（如 `293,398,530–532`）。这些不改变科学判断，但正式递交前必须清理并做干净环境 CI 编译。

## 8. 修订工作是否完整呈现在 reply

下表以当前 main diff、active `\rev` 与附录内容为依据：

| 实际修订/新增                                         | 稿件位置                                 | reply 呈现                    | 审核                                                           |
| ----------------------------------------------- | ------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| 补 QFT alternatives：Hales–Hallgren、Cleve–Watrous | `main:62–63`                         | 未在 summary/逐项中明确列出          | 遗漏；且该文献削弱 \(O(n^2)\) depth 叙事，应主动解释                          |
| 扩 HSP 重要性、解释 random shift                       | `main:79`                            | `reply:177,185,594–623`     | 已呈现                                                          |
| 解释 HP 架构动机                                      | `main:115–124`                       | `reply:601–614`             | 已呈现                                                          |
| 定义 \(\mathbb Z_{2^n}\) 与 Shor order \(r\)       | `main:140,376`                       | `reply:181,632–639`         | 已呈现；first appearance 仍不准确                                    |
| HP-0 从独立 section 降为一段并移附录                       | `main:171–176`; `appendix:329–439`   | `reply:172,241–251,560–563` | 已呈现但 `563` 错称完全 removed                                      |
| HP-0 新算法与 \(2^{-m}\) 保证                         | `appendix:397–439`                   | 只说 moved，未概述新结果             | 未充分呈现；若作为新增贡献应说明，但同时承认 classical baseline                    |
| 删除旧 HCR Eq. (8)                                 | `main:197–215` 注释                    | `reply:103,282–323,645–675` | 已呈现                                                          |
| 新 DFI/MLE 语境和引用                                 | `main:185–194`; `appendix:1233–1266` | `reply:150–160,295,475–522` | 已呈现，但引用适用范围和效率结论不自洽                                          |
| Theorem 2 / overlap analysis                    | `main:266–287`; `appendix:649–1160`  | `reply:103,325–347,652–675` | 已呈现但严重误呈现：称 analytical result，未披露 empirical closure/受限条件/负方向 |
| \(n=7–18\) DFI 回归                               | `appendix:1164–1207`                 | `reply:121–123,379`         | 已呈现，但把实际 \(n^2\) 窗写成 \(2^{n/2}\) “polynomial”                |
| 18q 网络结构、510 类、top-k                            | `appendix:1268–1343`                 | `reply:150–160,438–522`     | 部分；未说明 classifier-not-beam、指数输出头和全部训练成本                      |
| 全 period×shift 缓存与 held-out-shift split         | `appendix:1322–1330`                 | 未在 reply 清楚说明               | 遗漏；这决定 generalization 解释                                     |
| 18-bit semiprime benchmark                      | `main:351–380`                       | `reply:118,365–381`         | 已呈现；但未披露 base 后选择、denominator 和完整失败事件                        |
| 门级 Pauli noise、20 点曲线                           | `main:384`; `appendix:1360–1377`     | `reply:127–129,680–713`     | 已呈现；未呈现 \(s=0\)、4 trajectories、无 CI 和 51 CP 原因               |
| 新 limitation：无 closed-form decoder              | `main:389`; `appendix:1355`          | `reply:152,438–453`         | 已呈现，但标题/摘要未同步降 claim                                         |

结论：**“主要修改被提到”不等于“决定性限定被透明呈现”。** 当前 reply 的最大问题不是少列了几个小改动，而是把经验/有限/条件化的工作呈现成 analytical/scalable/efficient 的证据。

## 9. 修复优先级

### 9.1 P0：正式提交前必须停止并解决

1. 撤销或降格 Theorem 2；删除 positive growth 与 DFI→efficient sample complexity；
2. 把 exact dyadic theorem 与 generic truncated Shor states 明确分域；
3. 为 DFI 零分母给严格定义，并用同一指标重做 HP/QFT 比较；
4. 删除/证明 efficient poly decoder；修正 beam→logit ranking；
5. 删除/重做 general Shor replacement 与 base 后选择 benchmark；
6. 改正同一资源模型下的 HP/QFT depth/gate-count 比较；
7. 统一标题、摘要、Fig. 1、section title、summary 和 reply 的 claim。

### 9.2 P1：若保留强主张，必须新证明/重算/新实验

1. 所有 2-adic adjacent classes 的严格 overlap/error bound 与正指数结论；
2. 全局而非仅相邻 period 的 Chernoff/Hellinger/TV separation；
3. 输出、训练、推理、验证均 polynomial 的 decoder family；
4. unseen-period、unseen-\(n\)、多 seed 的独立 test；
5. 随机 base、完整失败事件、oracle/register/shot/GCD 成本的端到端 Shor；
6. 同硬件、同编译、同误差、同成功标准的 QFT baseline；
7. noise 多 trajectory/seed、shift 平均、CI 和 baseline。

### 9.3 P2：窄化路线也应完成

- 修 HP-L off-by-one、period windows、概率归一化、HSP 记号；
- 披露 phase optimization、随机曲线样本数、实际 phases/51 CP 规则；
- 修 figure captions、few epochs、monotonic、semiprime denominator；
- 精简附录重复推导；
- 修 reply 路径、labels、日期、源内工作注释和页/行定位。

## 10. 两条可执行改稿路线

### 路线 A：保留现有可靠成果，形成可信窄稿（推荐）

建议题目示例：

> **Shift-invariant Hadamard–phase circuits for dyadic hidden-subgroup sampling with finite-size learned decoding**

核心摘要应只说：

- 对 \(\mathbb Z_{2^n}\) 真子群证明 exact shift invariance；
- HP-0 是可解析 toy model，不主张量子优势；
- 对一般截断 periods 仅报告统一定义后的 finite-size divergence 与 \(n=18\) 510-class decoder；
- 不主张 polynomial decoder、general HSP 或 Shor/QFT replacement；
- Theorem 2 改 empirical conjecture/删去。

这条路线不需要把所有实验推倒重来，但需要全文系统性降 claim、修 DFI 定义和窗口，并如实改写 Shor 为 conditioned order-recovery benchmark。

### 路线 B：坚持原题目的强算法主张

最低交付应包括：

1. 对真实 Shor 截断陪集 \(|\psi_{r,c}\rangle\) 的 shift 误差统一界；
2. 对所有候选 periods 的全局分离和可达样本上界；
3. 不逐类枚举、具有端到端多项式复杂度的 decoder；
4. 按随机 base 与标准 inclusion rule 的端到端 factoring 成功率；
5. 完整寄存器、oracle、shots、验证与失败成本；
6. 同一物理/编译模型下 exact/approximate QFT 基线；
7. 跨 \(n\)、未见 periods、独立 test 与统计不确定性。

这实际上是新的理论与实验项目，而不是现有 reply 的文字修订。

## 11. reply 建议结构

每条审稿意见使用固定三段：

1. **Position**：明确 agree/disagree；
2. **Change and evidence**：给修改后的确切句子、主文/附录页行和新证据；
3. **Remaining limitation**：明确仍未证明什么，并保证标题/摘要同步。

尤其 Referee A3 可改为类似：

> We agree and withdraw the claim of efficient sample complexity. We now use \(\chi^2(P_{r+1}\Vert P_r)\) only as a finite-size diagnostic of adjacent-period distinguishability. The overlap relation is an empirical ansatz; conditional on it, the calculation gives \(\Omega(1/r)\) for a restricted 2-adic class, not a positive-growth theorem or an achievable sample bound. We have therefore relabeled the result and removed “efficient” from the title, abstract, and conclusions.

这比当前一面承认无上界、一面继续暗示 efficient 更可信。

## 12. 最终判断

Referee A 的核心批评仍成立，而且新版 Theorem 2 与 decoder 披露反而增加了新的 validity concern。Referee B 的 notation、作图和组织意见大多可以解决，但 \(\eta\) rationale 仍缺。Editor 已明确拒稿并否定 PRL 综合门槛；当前 reply 没有足以逆转这一决定的闭环证据。

最有建设性的结论不是“项目失败”，而是：**目前存在一篇范围更窄、可诚实成立的论文，但不存在现标题所宣称的已证高效 QFT/Shor 替代方案。** 先决定走窄化路线还是强主张路线，再改 reply；否则继续在同一版中同时保留让步与强结论，只会再次触发相同拒稿理由。

## 13. 审查完成度与交付说明

本报告是在六个问题各自完成“支持方—反对方”初审、交换论证后的第二轮交叉、主审独立数值/图形/编译核验之后定稿；审查窗口按用户要求超过两小时。报告没有声称执行代码仓库的完整运行审计：压缩包只提供了本文所列 TeX/图形/参考文献材料，因此关于生成脚本、真实硬件执行和随机种子复现的判断均明确标为“未披露/需补充”，而不是把缺少的证据推断成已验证结果。
