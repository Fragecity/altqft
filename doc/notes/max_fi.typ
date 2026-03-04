#import "@preview/quill:0.7.2": *
#import "@preview/physica:0.9.8": *
= SINV circuit

we proof that a circuit with the following construction is shift invariant:

#let qc = quantum-circuit(
    1, $H$, ctrl(1), ctrl(3), [\ ],
    2, ctrl(), 2, ctrl(), $H$, [\ ], 
    1, $H$, 2, ctrl(1), ctrl(-1), [\ ],
    3, ctrl(), ctrl(), 1,$H$,1
  )
#{


  let qc2 = quantum-circuit(
    1, $H$, ctrl(1), ctrl(2), [\ ],
    2, ctrl(), 3, ctrl(), $H$, [\ ], 
    3,ctrl(),$H$, ctrl(1), ctrl(-1), [\ ],
    5, ctrl(),1,$H$,1
  )

  figure(grid(
    columns: 2,
    gutter: 3em,
    qc, qc2
  ))
}

The feature of this cirucit is:
- there is only one Hadamard gate on each qubit
- control phase gate $"CP"(i,j)$ only appear once for each $(i,j)$
- for each CP gate, there are one H on left and one H on right

== proof sketch
the circuit with above structure will map $ket(b)$ to 
$
  ket(b_1 b_2 ...) -> sum_(c_i in {0,1}) exp(sum_(j,k) theta_(j,k) b_j c_k) ket(c_1 c_2 ...)
$ 
when we consider group $ZZ_(2^n)$, the subgroup obtain
$
  H = {b_1 b_2 ... b_m 0 ... 0 | b_i in {0,1} }
$
and the coset is 
$
  c+H = {b_1 b_2 ... b_m b'_(m+1) ... b'_n | b_i in {0,1} }
$

thus, 

$
  braket(g, U, c+H) = exp(sum_(j,k) theta_(j,k) b'_j c_k) sum_(b_i)  exp(sum_(k)sum_(1<=j<=m) theta_(j,k) b_j c_k)
$


== things that interesting

Its not only QFT is SINV, but the above circuit.


= Fisher Information

now we consider circuit with ansatz

#qc

The interlace hih circuit.

we let the phase be $pi/2^"d" + theta$, and we train the parameter $theta$. We have
#figure(image("../../fig/loss_landscape.png"))

the cost function is $-"FI"$. So the NN aims to maximize the fisher info. The result shows that, the FI does not decrease a lot in training. 

So we consider the initial parameter as a relatively good choice of parameter for FI. And we care about how FI goes up when qubits goes up. The numerical result shows that:
#figure(image("../../fig/init_cost_landscape.png"))

Surprising! It seems that the FI goes linearly!

It seems that the new circuit is tentitive of period!

== A difference with former result

In former result, the circuit is 
#{


  let qc2 = quantum-circuit(
    1, $H$, ctrl(1), ctrl(3), [\ ],
    2, ctrl(), 2, $H$, [\ ], 
    1, $H$, 2, ctrl(1),1, [\ ],
    3, ctrl(), ctrl(), $H$,1
  )

  figure(qc2)
}
The key thing is, the first qubit do not have any info about the third qubit! I guess it is a big reason why the FI do not perform well on that ansatz.


= Machine learning
we apply U on state $ket(c + H)$, and measure it $n^2 * 1024$ times. the result is 

#figure(image("../../fig/counts.png"))

we can zoom the non-zero area

#figure(image("../../fig/counts_decimal_filtered_plot.png"))
 you can see that the frequency carries information, since it is not "uniform".

so, I use the frequency and a and N to train the NN

#pagebreak()

== NN set-up
- data: x = (counts, a, N)
- target: y = r s.t. $a^r = 1 mod N$.
- nn: $z_(i+1) = "relu"(W_i z_(i))$
- cost: cross entropy with real $r$

== NN problem
+ exponential size of nn
+ cost exponential resource to calculate $r$
+ size of counts, a and N do not match
+ the structure of nn is too simple

== NN result
#figure(image("../../fig/test_accuracy_plot.png"))

== Ablation

to test if the nn actual use the quantum measurement result to calculate the result, instead of cheating: directly calculate $r$. We perturbate the measurement results. If the nn cheats, the results should not depend on the measurements. 

#figure(image("../../fig/robustness_plot_shuffle.png"))