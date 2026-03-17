#import "@local/mynote:0.1.0": msym, note, thm
#import thm: *
#import msym: *

#import "@preview/quill:0.7.2": *
#import "@preview/mitex:0.2.6": *

#show: note.with()

#definition("suspicious circuit")[
  a unitary with $U_(i j) = exp(ii theta_(i j))$ is called suspicious unitary.
]
#remark[ not all suspicious circuit is shift invariant, since I found a counter example]

#definition("hi circuit")[
  a circuit with one hadamard gate on each qubit, and free phase block.
]

#remark[ not all hi circuit circuit is shift invariant, since I found a counter example]

// #question[Given a SINV gate A, and phase gate B. We want $(bb(1)_A tensor H) B (A tensor bb(1)_2 )$ is SINV. What is the condition of B?]
// #quantum-circuit(
//     1, nwire(""), gate($A$), mqgate($B$, n: 2), [\ ],
//     4, $H$, 1
// )

#lemma[
  2q hi circuit is SINV.
]
#figure(quantum-circuit(
  1,
  $H$,
  ctrl(1, wire-label: $theta$),
  1,
  [\ ],
  2,
  ctrl(),
  $H$,
  1,
))
The circuit is SINV for all $theta$ #sym.checkmark

#remark[its seems free for 3 qubits]
#figure(quantum-circuit(
  1,
  $H$,
  ctrl(1, wire-label: $theta_1$),
  2,
  ctrl(2, wire-label: $theta_3$),
  1,
  [\ ],
  2,
  ctrl(),
  $H$,
  ctrl(1, wire-label: $theta_2$),
  [\ ],
  4,
  ctrl(),
  ctrl(),
  $H$,
  1,
))
The above circuit is
#mitex(
  `\left[\begin{matrix}1 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & e^{i \theta_{3}} & e^{i \theta_{1}} & e^{i \left(\theta_{1} + \theta_{3}\right)} & -1 & - e^{i \theta_{3}} & - e^{i \theta_{1}} & - e^{i \left(\theta_{1} + \theta_{3}\right)}\\1 & e^{i \theta_{2}} & -1 & - e^{i \theta_{2}} & 1 & e^{i \theta_{2}} & -1 & - e^{i \theta_{2}}\\1 & e^{i \left(\theta_{2} + \theta_{3}\right)} & - e^{i \theta_{1}} & - e^{i \left(\theta_{1} + \theta_{2} + \theta_{3}\right)} & -1 & - e^{i \left(\theta_{2} + \theta_{3}\right)} & e^{i \theta_{1}} & e^{i \left(\theta_{1} + \theta_{2} + \theta_{3}\right)}\\1 & -1 & 1 & -1 & 1 & -1 & 1 & -1\\1 & - e^{i \theta_{3}} & e^{i \theta_{1}} & - e^{i \left(\theta_{1} + \theta_{3}\right)} & -1 & e^{i \theta_{3}} & - e^{i \theta_{1}} & e^{i \left(\theta_{1} + \theta_{3}\right)}\\1 & - e^{i \theta_{2}} & -1 & e^{i \theta_{2}} & 1 & - e^{i \theta_{2}} & -1 & e^{i \theta_{2}}\\1 & - e^{i \left(\theta_{2} + \theta_{3}\right)} & - e^{i \theta_{1}} & e^{i \left(\theta_{1} + \theta_{2} + \theta_{3}\right)} & -1 & e^{i \left(\theta_{2} + \theta_{3}\right)} & e^{i \theta_{1}} & - e^{i \left(\theta_{1} + \theta_{2} + \theta_{3}\right)}\end{matrix}\right]`,
)


#let qceq = {
  let qc1 = quantum-circuit(
    1,
    $H$,
    ctrl(2),
    2,
    [\ ],
    3,
    $H$,
    [\ ],
    1,
    $H$,
    ctrl(),
    2,
  )

  let qc2 = quantum-circuit(
    1,
    $H$,
    ctrl(2),
    1,
    [\ ],
    1,
    $H$,
    [\ ],
    1,
    $H$,
    ctrl(),
    1,
  )

  let qc3 = quantum-circuit(
    1,
    $H$,
    1,
    [\ ],
    1,
    $H$,
    [\ ],
    1,
    $H$,
    1,
  )
  
  figure(grid(
    align: center+horizon,
    inset: 4pt,
    columns: 5,
    qc1,$=$, qc2, $approx$, qc3 
  ))
}

#qceq


