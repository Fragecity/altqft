from __future__ import annotations


def build_default_period_range(nqubit: int) -> list[int]:
    if nqubit < 2:
        raise ValueError("nqubit must be at least 2")

    upper_bound = min(2**nqubit, max(nqubit**2, int(2 ** (nqubit / 4))))
    periods = list(range(nqubit, upper_bound))
    if not periods:
        raise ValueError(f"no valid period range for nqubit={nqubit}")
    return periods
