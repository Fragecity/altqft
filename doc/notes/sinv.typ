#import "@local/mynote:0.1.0": msym, note, thm
#import thm: *
#import msym: *

#import "@preview/physica:0.9.8": *
#import "@preview/quill:0.7.2": *
#import "@preview/mitex:0.2.6": *
#import tequila as tq

#show: note.with()

#let cp = $"CP"$
#let pqft = "pQFT"

= 123

= Parameterized QFT
#definition("parameterized QFT")[
  The circuit
  $
    pqft(bold(theta)) = product_(k = 1)^n H_k product_(j>k) cp(k, j, theta_(k,j))
  $
  is called parameterized QFT
]

#let pqft-qc(nqubit: 3) = {
  let h-cp(i) = tq.build(
    x: 1 + int(i * (2 * nqubit - i + 1) / 2),
    tq.h(i),
    for qubit in range(i + 1, nqubit) {
      tq.ca(i, qubit, $R_z (theta_(#i,qubit))$)
    },
  )

  quantum-circuit(
    row-spacing: 5pt,
    column-spacing: 5pt,
    // ..h-cp(1), ..h-cp(2)
    ..for i in range(nqubit) {
      (..h-cp(i),)
    },
  )
}


#figure(grid(
  align: center + horizon,
  columns: 3,
  $pqft(bold(theta))$, $=$, pqft-qc(),
))

#theorem[
  pQFT is shift invariant under group $Z_(2^n)$ for all $bold(theta)$.
]
#proof[
  After applying $pqft(bold(theta))$ on state $ket(b_(n-1) b_(n-2) dots b_0)$ we have

  $
      & pqft(bold(theta))ket(b_(n-1) b_(n-2) dots b_0) \
    = & (ket(0) + exp(ii pi b_0) ket(i)) kron (ket(0) + exp(ii pi b_1 + theta_(0,1)b_0) ket(i)) kron ... \
      & kron (ket(0) + exp(ii pi b_(n-1) + sum_(j< n-1) theta_(j,n-1)b_j) ket(i)) \
    = & sum_(c_i in {0,1}) exp[
          ii sum_(i = 0)^(n-1) c_i (pi b_r(i) + sum_(j<r(i)) theta_(j, r(i)) b_j)
        ] ket(c_(n-1) c_(n-2) dots c_0) \
    = & sum_(c_i in {0,1}) exp[
          ii sum_(i = 0)^(n-1) c_i f_b (i)
        ]ket(c_(n-1) c_(n-2) dots c_0).
  $
  Here, $r(i)$ represent the reverse of qubits, as what we do in QFT, $r(i) = n-i-1$. The $f_b (i)$ represent the phase
  $
    f_b (i) = pi b_r(i) + sum_(j<r(i)) theta_(j, r(i)) b_j.
  $<eq:lemma1-2>

  A subgroup of $ZZ_(2^n)$ is
  $
    H = {b_(n-1) b_(n-2) dots b_(m) 0 ... 0 | b_i in {0,1}},
  $
  and the coset is
  $
    bold(b)' + H = {b_(n-1) b_(n-2) dots b_(m) b'_(m-1) ... b'_0 | b_i in {0,1}}
  $
  The shift invariant condition is
  $
    | braket(bold(c), pqft(bold(theta)), H) |^2 = | braket(bold(c), pqft(bold(theta)), bold(b)' +H) |^2, #h(1em) forall bold(c), bold(b)', forall H< Z_(2^n) ,
  $<eq:lemma1-1>
  where $ket(H) = sum ket(b_(n-1) b_(n-2) dots b_(m) 0 ... 0)$.
  This condition is automatically satisfied since @eq:lemma1-2 is a linear combination of $b_i$. We can always separate the "free bits" $b_i$ with "froze bits" $b'_i$. 
  // Now, let's check if @eq:lemma1-1 is correct.

  // We split the "free bits" with "froze bits",
  // $
  //     & braket(bold(c), pqft(bold(theta)), bold(b)' +H) \
  //   = & sum_(b in bold(b)' +H) exp[
  //         ii sum_(i = 0)^(n-1-m) c_i f_b (i) + ii sum_(n-m)^(n-1) c_i f_b (i)
  //       ]
  // $
  // Notice that $f_b (i)$ only contains fix bits $b'$ when $i >= n-m$ since 
  // $
  //   j < r(i) <= m-1
  // $
  // and $b_k = b'_k$ when $k <= m-1 $.

  // When $ 0 <= i < n-m$,   
  // $
  //   f_b (i) =& (pi b_r(i) + sum_(m <= j<r(i)) theta_(j, r(i)) b_j) + sum_(0<=j <m)theta_(j, r(i)) b_j \
  //   = & f_(b,1)(i) + f_(b,2)(i)
  // $

]

#corollary[spQFT circuit are shift invariant]
