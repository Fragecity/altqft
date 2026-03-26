import pytest

from altqft.nn.periods import build_default_period_range


@pytest.mark.parametrize(
    ("nqubit", "expected"),
    [
        (2, [2, 3]),
        (3, [3, 4, 5, 6, 7]),
        (4, list(range(4, 16))),
    ],
)
def test_build_default_period_range_uses_half_open_interval(
    nqubit: int,
    expected: list[int],
) -> None:
    assert build_default_period_range(nqubit) == expected
    assert max(expected) < 2**nqubit
