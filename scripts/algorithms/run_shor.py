from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from altqft.algorithms import ShorConfig, default_counting_qubits, run_shor

DEFAULT_N = 15
DEFAULT_A = 2
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 7


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small-scale Shor factoring demo with Qiskit statevector sampling.",
    )
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Composite integer to factor.")
    parser.add_argument(
        "--a",
        type=int,
        default=DEFAULT_A,
        help="Coprime base used for modular order finding.",
    )
    parser.add_argument(
        "--counting-qubits",
        type=int,
        default=None,
        help="Number of counting-register qubits. Defaults to 2 * ceil(log2(N)).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Number of sampler shots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampler seed.",
    )
    return parser.parse_args(argv)


def resolve_counting_qubits(args: argparse.Namespace) -> int:
    if args.counting_qubits is not None:
        return int(args.counting_qubits)
    return default_counting_qubits(int(args.N))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ShorConfig(
        N=int(args.N),
        a=int(args.a),
        counting_qubits=resolve_counting_qubits(args),
        shots=int(args.shots),
        seed=int(args.seed),
    )
    result = run_shor(config)

    print(
        "config "
        f"N={config.N} "
        f"a={config.a} "
        f"counting_qubits={config.counting_qubits} "
        f"work_qubits={math.ceil(math.log2(config.N))} "
        f"shots={config.shots} "
        f"seed={config.seed}"
    )

    gcd_value = math.gcd(config.a, config.N)
    if 1 < gcd_value < config.N and result.factors is not None:
        print(f"gcd shortcut factor={result.factors[0]} other_factor={result.factors[1]}")

    for candidate in result.candidates:
        print(
            "candidate "
            f"bitstring={candidate.bitstring} "
            f"count={candidate.count} "
            f"phase={candidate.phase:.6f} "
            f"denominator={candidate.continued_fraction_denominator} "
            f"validated_order={candidate.validated_order} "
            f"status={candidate.status}"
        )

    if result.success and result.factors is not None:
        if result.order is not None:
            print(f"validated order r={result.order}")
        print(f"{config.N} = {result.factors[0]} x {result.factors[1]}")
        return 0

    print(f"failed to recover non-trivial factors for N={config.N} with a={config.a}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
