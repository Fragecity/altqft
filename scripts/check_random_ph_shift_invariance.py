from __future__ import annotations

import random

import numpy as np
from qiskit import QuantumCircuit

from altqft.circuits.ph import ph_phase
from altqft.nn.process_qc import make_prob
from fisher_information_utils import random_hlayout


def is_shift_invariant(circuit: QuantumCircuit, col: int, period: int) -> bool:
    probability = make_prob(circuit, period)
    baseline = probability(col, 0)
    return all(np.isclose(probability(col, shift), baseline) for shift in range(1, period))


def check_random_ph_shift_invariance(nqubits: int, seed: int | None = None) -> tuple[list[int], bool]:
    rng = np.random.default_rng(seed)
    hlayout = random_hlayout(nqubits, rng=random.Random(seed))
    circuit = ph_phase(hlayout)
    col = int(rng.integers(0, 2**nqubits))
    period = 2
    return hlayout, is_shift_invariant(circuit, col=col, period=period)


def main() -> None:
    nqubits = 4
    hlayout, result = check_random_ph_shift_invariance(nqubits, seed=7)
    print(f"随机 PH 线路 hlayout={hlayout} 是否具有 shift invariant: {result}")


if __name__ == "__main__":
    main()
