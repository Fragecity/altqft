#import "@local/note:0.1.0": *
#import "@preview/quill:0.7.1": *
#import "@preview/equate:0.3.2": equate

#show: template.with(name: "Shift Invariance")

#show: equate.with(breakable: true, sub-numbering: true)
// #set math.equation(numbering: n => {
//   numbering("(1.1.1)", counter(heading).get().first(), n)}
// )
#set math.equation(numbering: "(1.1)")
#let CP = $op("CP")$
#let qft = $op("QFT")$
#let ee = $sans(e)$
#let circuit-equation(A, B) = align(center, grid(
  columns: 3,
  align: center + horizon,
  A, $=$, B,
))

= Motivation
Our goal is to extend the QFT framework to encompass a broader class of gates, particularly those that exhibit *shift invariance*. By exploring this expanded gate set, we aim to identify gates that are practically implementable on current quantum hardware.

= Overview
*2025-9-5*
1. Discuss the efficiency of the 
= Shift Invariance
== Shift Invariance in QFT
To understand shift invariance, we begin with the QFT:
$ qft ket(c h) = 1 / sqrt(G) sum_g rho(c h) ket(g) $

Here, $c$ represents a coset label, and $rho$ is a specific representation. When applied to a superposition state $sum_h ket(c h)$, the QFT yields:
$ qft sum_(h in H) ket(c h) = rho(c) / sqrt(G) sum_(h g) rho(h) ket(g) $

Our goal is to eliminate the influence of the coset label $c$ in the QFT, retaining only the information about the subgroup $H$.

To extract this information, we measure the quantum register. Removing the effect of $c$ is equivalent to ensuring:
$ abs( rho(c_1) / sqrt(G) sum_h rho(h)) = abs(rho(c_2) / sqrt(G) sum_h rho(h)) $

This equality holds because $rho(c_1)$ and $rho(c_2)$ differ only by a phase factor. While not a global phase, this ensures that the probability of measuring $ket(g)$ remains the same.





= Z Transform
We begin by analyzing the QFT and then explore how the classical Z transformation can be extended to its quantum counterpart.

== Discrete Fourier Transform
A data vector can be expressed as:
$ bold(x) = sum x_i e_i $
where $e_i$ represents the natural basis:
$ e_i = [0, ..., 0, 1, 0, ..., 0] $
(1 in the $i$-th position).

We perform a basis transformation $e_i -> f_i$, where:
$ f_k = 1 / sqrt(N) sum ee^(2 pi ii  j  k / N)  e_j $

This gives:
$ bold(x) = sum x_i  e_i = sum y_j  f_j $

and:
$ y_j = 1 / sqrt(N)  sum ee^(-2  pi  i  j  k / N)  x_k $

We verify this transformation as follows:
$ x_l =& e_l^T  sum y_k  f_k = 1 / N  sum_(j, k) ee^(-(2  pi  ii  j  k) / N)  x_j  ee^((2  pi  ii  l  k) / N) \ 
 =& 1 / N  sum_j sum_k  ee^((-2  pi  ii  (j - l)  k) / N)  x_j = x_l $

If we use $ket(j)$ to represent the basis $e_j$, then $e_j-> f_j$ could be written as 
$ ket(k) -> 1 / sqrt(N) sum ee^((2 pi ii j k) / N) ket(j) $
Thus, from this aspect, we can see that the QFT is essentially a basis transformation.
This establishes the quantum analog of the classical Fourier Transform. 

== Z transformation

We gain insights from the QFT when attempting to "quantize" a classical transformation. Specifically, we consider Z transformation as a kind of basis transformation. And analyze how the Z transformation interacts with the basis vectors $e_i$.

For the Z transformation, the operation $cal(Z)$ acts on the basis $e_k$ as follows:
$
  cal(Z): e_k -> sum_j j^(-k)e_j
$
Then, we have the quantum version of Z transformation:
$
  ket(k) -> sum_j j^(-k) ket(j).
$

If Z transformation satisfy shift invariance?
$
  cal(Z) ket(c + H) = sum_h sum_j j^(-(c+h)) ket(j) = sum_j j^(-c) (sum_h j^(-h)) ket(j).
$
The probability to get $j$ is with different coset is $ abs(j^(-c_1) sum j^(-h))$ and $abs(j^(-c_2) sum j^(-h))$. It still carries the information of $c$.
Do not satisfy shift invariance.

= General Shift Invariance
== Shift operation
For simple, we consider abelian group $G$ first.

Let $T$ be the operation that do the shift, and $ket(H) = sum ket(h)$
$
  T_c ket(g) = ket(g +c) ,forall g in G.
$
Then the shift invariant condition could be expressed as
$
  |bra(g)qft T_c ket(H)|^2 = |bra(g)qft T_(c') ket(H)|^2, space.quad forall g, c in G
$

== Formal definition of Shift Invariance

Notice: I didn't find a 'formal' definition of shift invariance in the literature. However, we can define it as follows:
#definition("Shift Invariance (over abelian group)")[
  Given an abelian group $G$ and any subgroup $H$, a unitary operation $U$ is said to be shift invariant if
  $
    |bra(g)U ket(H)|^2 = |bra(g)U T_(c) ket(H)|^2, space.quad forall g, c in G
  $
]
== Equivalent condition
$
  &|bra(g)U ket(H)|^2 = |bra(g)U T_(c) ket(H)|^2, space.quad forall g, c in G \
  <==> & angle.l angle.l g | cal(U) | H angle.r angle.r = angle.l angle.l g | cal(U) cal(T)_c | H angle.r angle.r, space.quad forall g, c in G \
  <==> & angle.l angle.l g | cal(U) | 0 angle.r angle.r = angle.l angle.l g | cal(U) | k angle.r angle.r, space.quad forall k in G \
  <==> & bra(g) U ketbra(0) U^dagger ket(g) = bra(g) U ketbra(j) U^dagger ket(g) , forall j,g in G \
  <==> & |U_(0,g)|^2 = |U_(k,g)|^2, forall k,g in G
$

We know that $sum_i |U_(i,g)|^2 = 1$. Thus,
$
       & |U_(0,g)|^2 = |U_(k,g)|^2, forall k,g in G \
  <==> & |U_(i,g)|^2 = 1/sqrt(N)
$

Finally, we have
$
  & U "satisfy shift invariance" \
  <==> & U = 1/sqrt(N)mat(e^(ii theta_11), ..., e^(ii theta_(1N)); dots.v, space, dots.v; e^(ii theta_(N 1)), ..., e^(ii theta_(N N)))
$<eq:1>

= shift-invariance gate and Hadarmard-Ising gates
One example of shift-invariance gate is QFT, which includes $H^(otimes n)$.

$
  H^(otimes n) = 1/sqrt(N) mat(plus.minus 1, ..., plus.minus 1; dots.v, space, dots.v; plus.minus 1, ..., plus.minus 1)
$
and Ising gates
$
  "Is"(bold(theta)) = & exp(ii sum theta_(i j) Z_i Z_j + ii sum theta_(i) Z_i) \
                    = & mat(e^(ii phi_11), space, space; space, dots.down, space; space space, space, e^(ii phi_(N N)))
$
Hadamard-Ising gates $"Is"(bold(theta)) H^(otimes n)$ satisfy @eq:1, which means they satisfy the shift invariance.

= What happen if we use Hadamard-Ising gates instead of QFT in Fourier sampling?

After measuring a quantum data table, we get $sum_h ket(c+h)$. Then, apply Hadamard-Ising gates
$
  "Is"(bold(theta)) H^(otimes n) sum_h ket(c+h) = sum_k e^(ii sum_(i j) k_i k_j theta_(i j)) (-1)^(bold(k)dot bold(c)) ket(0)^(otimes(n-p))ket(k)
$<eq:2>

#rect[
  $
    H^(otimes n) ket(g_1g_2 ... g_n) = otimes (ket(0) + (-1)^(g_j) ket(1))
    = sum (-1)^(sum k_j g_j)ket(k)
  $

  $
    H^(otimes n) sum_h ket(c+h) = sum_h sum_k (-1)^(sum k_j (c+h)_j)ket(k)
  $
  Suppose $G tilde.eq ZZ_(2^m)$, the element of $H$ must be written by $q 2^p$, where $p,q in ZZ^+$. It means that the binary format of $h$ is
  $
    h = h_1 h_2 ... h_(n-p) 0 ... 0.
  $
  //   $h_1 ... h_(n-p)$ can go through $0, 1$. Thus, $(h+c)_j$ is $0$ or $1$ or go through ${0,1}$. Select the sequence that $(h+c)_j$ are fix
  //   $
  //     h+c = g_1 ... g_(r-1) d_r ... d_(r+s) g_(r+s+1) ...g_n
  //   $
  //  When $h$ go through $H$, $g$ go through ${0, 1}$, and
  without loss of generality, let $c = 0..0 c_(n-p+1)...c_n$. Thus,
  $
    h + c = h_1 h_2 ... h_(n-p) c_(n-p+1)...c_n
  $
  Thus, $sum_h (-1)^(sum k_j (c+h)_j) eq.not 0$ implies $k_1 = k_2 = ... = k_(n-p) = 0$. For convience, denote $bold(k)dot bold(c) = k_(n-p+1)c_(n-p+1) + ... + k_n c_n$. Then
  $
    H^(otimes n) sum_h ket(c+h) = sum_k (-1)^(bold(k)dot bold(c)) ket(0)^(otimes(n-p))ket(k)
  $
  And then apply Ising gates
  $
    "Is"(bold(theta)) H^(otimes n) sum_h ket(c+h) = sum_k e^(ii sum_(i j) k_i k_j theta_(i j)) (-1)^(bold(k)dot bold(c)) ket(0)^(otimes(n-p))ket(k)
  $

]

== Simple example
$
  G tilde.eq ZZ_(4) = {0,1,2,3}\
  G>H = {0,2} = {00, 10}
$
Then, apply H:
$
  &H^(otimes 2) ket(H) = H^(otimes 2) (ket(00) + ket(10))\
  =& (ket(00) + ket(01) + ket(10) + ket(11)) 
  + (ket(00) + ket(01) - ket(10) - ket(11)) \
  =&  ket(00) + ket(01)
$
So after sampling, we know the first bit is always 0, which means $H = {0,2}$

Compare to the QFT
$
  &qft ket(H) \
  =& (ket(00) + ket(01) + ket(10) + ket(11)) 
  + (ket(00) - ket(01) + ket(10) - ket(11)) \
  =&  ket(00) + ket(10)
$

*SURPRISE!!!*  If $G tilde.eq otimes ZZ_(2^(m_i))$, it seems that the $H^(otimes n)$ gate is enough!

== Efficiency discussion
The algorithm of finding hidden subgroup with $otimes ZZ_(2^m_i)$ with Hadamard gates is:
 
+ apply $H^(otimes n)$
+ measure the register $m$ times
+ according to the measurement results, for each bit, decide the bit is always 0, or it is random (it's 0 or 1).

After measurement, denote the measurement result as ${bold(b)^((i))}_(i=1)^m$, and $bold(b)^((i)) = b^((i))_1 b^((i))_2...b^((i))_n$. For each bit $b_j$, we need to decide if it is always 0, or it is random (it's 0 or 1). The probability that we make wrong decision (the bit is random but we think it is always 0) is $(1/2)^m$. 

Thus,
$
  "Prob"("right decision") >= (1 - 1/2^m)^n >= 1 - n/2^m >= 1-epsilon 
$
leads to 
$
  n/2^m <= epsilon \ 
  m >= log_2(n / epsilon)
$

We conclude that, when $m >= log_2(n / epsilon)$, the probability of right decision is at least $1 - epsilon$.

== Box 5.4 Factoring 15 quantum-mechanically (page 235 in Nielsen)

The equation (5.62) keep the same. 
$
  & 1/sqrt(2^t) sum_(k=0)^(2^t-1) ket(k) ket(x^k "mod" N)
  =& 1/sqrt(2^t) [ket(0)ket(1) + ket(1)ket(7) + ket(2)ket(4) + ...]
$
measure the second register, suppose we get $ket(4)$. The first register is 
$
  ket(2) + ket(6) + ket(10) + ...
$
applying Hadamard gates, we have $...00, ...01, ...10, ...11$. Thus, we get $r = 4$, $7^4 "mod" 15 = 1$. 

The rest part follows the box 5.4.


= Advantage of QFT

From the above section, we found that for certain scenario, the Hadamard gates are enough for finding hidden subgroup. However, the QFT has it's own advantage: *When the given group $G$ is not $ZZ_2^(m_i)$, it can use a $ZZ_2^(m_i)$ to approximate it*. 

Let's illustrate this with an example.

Suppose $G tilde.eq ZZ$, $G>H tilde.eq ZZ "/" ZZ_7$, 
$
  G = {..., -3,-2,-1,0,1,2,3,...}, \
  H = {... -14, -7, 0, 7, 14, ... } < G .
$
And a group function $f: G -> X$, with hidden subgroup $H$ (we do not care what the funtion looks like, but we know that $f(x) = f(x+7)$). 
Then, we can *"trancate"* the group $G$ with a $ZZ_(2^n)$, say $ZZ_(2^10)$. And limit the function $f$ to the truncated group $ZZ_(2^10)$. 
Notice that the *trancated subgroup $H' = {0, 7, ...}$ is not a subgroup of trancated group $ZZ_(2^10)$*. 

*However, the QFT can still approximately find the hidden subgroup in this case!*

#block(stroke: black, inset: 10pt)[
  $
    qft^(-1) ket(H') = sum_k (sum_(j=0)^146 exp(- (2pi ii dot 7j dot k )/2^10)  ) ket(k)
  $
  Then, we print the unnormalized factor of the first 10 kets with largest probability:
  #table( columns: 5,
    $ket(0): 147$, $ket(146): 80 + 100 ii$, $ket(147): -31 - 40 ii$, $ket(292): -17 + 77ii$, $ket(293): 24-103 ii$,
    $ket(439): 128 -61ii$, $ket(585): 128 + 61ii$, $ket(731): 24+103 ii$, $ket(732): -17-77 ii$ , $ket(878): 80-99 ii$ 
    )

  #image("../../qft_approx.png", width: 80%)

  Apply continued fraction:
  $
    146/1024 = 1/(6 + 1/(1+ ...)) \
  $
  In the first iteration, we check that if $f(x+ 6) = f(x)$. It fails.
  In the second iteration, $1/(6 + 1/1) = 1/7$, we check that if $f(x+7) = f(x)$. It success. 

  We can also try some other ket, like $ket(585)$
  $
    585/1024 = 1/(1 + 1/(1+ 1/(3 + 1/146))) \
  $
  when we check the fraction in third iteration, $1/(1 + 1/(1 + 1/3)) = 4/7$, $f(x+7) = f(x)$. It success.
]

