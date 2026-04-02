from __future__ import annotations

from collections.abc import Sequence


def default_period_upper_bound(nqubit: int) -> int:
    if nqubit < 2:
        raise ValueError("nqubit must be at least 2")
    return min(2**nqubit, max(nqubit**2, int(2 ** (nqubit / 4))))


def build_period_range(
    nqubit: int,
    *,
    min_period: int = 2,
    max_period: int | None = None,
) -> list[int]:
    upper_bound = default_period_upper_bound(nqubit)
    resolved_max = upper_bound - 1 if max_period is None else max_period

    if min_period < 2:
        raise ValueError("min_period must be at least 2")
    if resolved_max >= 2**nqubit:
        raise ValueError("max_period must be smaller than the computational basis size")
    if resolved_max < min_period:
        raise ValueError("max_period must be greater than or equal to min_period")

    periods = list(range(min_period, resolved_max + 1))
    if not periods:
        raise ValueError(
            f"no valid period range for nqubit={nqubit}, min_period={min_period}, max_period={resolved_max}"
        )
    return periods


def build_legacy_period_range(nqubit: int) -> list[int]:
    return build_period_range(
        nqubit,
        min_period=nqubit,
        max_period=default_period_upper_bound(nqubit) - 1,
    )


def build_default_period_range(nqubit: int) -> list[int]:
    return build_period_range(nqubit)


def period_range_tag(period_range: Sequence[int]) -> str:
    if not period_range:
        raise ValueError("period_range must not be empty")
    return f"p{int(period_range[0])}-{int(period_range[-1])}"


def period_range_artifact_suffix(nqubit: int, period_range: Sequence[int]) -> str:
    return "" if list(period_range) == build_legacy_period_range(nqubit) else f"_{period_range_tag(period_range)}"
