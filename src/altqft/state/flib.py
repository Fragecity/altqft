from __future__ import annotations


def _modular_value(a: int, x: int, modulus: int) -> int:
    return 1 if x == 0 else pow(a, x, modulus)


def find_solutions(a: int, c: int, N: int, n: int) -> list[int]:
    if N == 1:
        if c != 0:
            raise ValueError("c must be 0 when N == 1")
        return list(range(1 << n))

    solutions = [x for x in range(1 << n) if _modular_value(a, x, N) == c]
    if not solutions:
        raise ValueError(f"No solution found for a={a}, c={c}, N={N}")

    return solutions
