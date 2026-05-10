from __future__ import annotations

from pathlib import Path

import pytest

from altqft.algorithms.shor_ph1 import (
    PH1ShorConfig,
    _collapsed_periodic_state,
    _modular_exponentiation_outputs,
    run_shor_with_ph1,
)


def test_collapsed_periodic_state_has_expected_support_for_15() -> None:
    config = PH1ShorConfig(N=15, a=2, nqubit=4, measurement_count=128, top_k=3, seed=7)
    outputs = _modular_exponentiation_outputs(config)

    state, support = _collapsed_periodic_state(config, outputs=outputs, work_value=1)

    assert support == (0, 4, 8, 12)
    probabilities = state.probabilities()
    assert probabilities[0] == pytest.approx(0.25)
    assert probabilities[4] == pytest.approx(0.25)
    assert probabilities[8] == pytest.approx(0.25)
    assert probabilities[12] == pytest.approx(0.25)


def test_run_shor_with_ph1_factors_15_with_repo_artifacts() -> None:
    config = PH1ShorConfig(
        N=15,
        a=2,
        nqubit=4,
        period_min=4,
        period_max=15,
        measurement_count=16_384,
        top_k=3,
        seed=7,
        model_dir=Path("model"),
        data_dir=Path("data"),
        output_dir=Path("outputs"),
    )

    result = run_shor_with_ph1(config)

    assert result.success
    assert result.top1_period == 10
    assert result.predicted_period == 12
    assert result.factors == (3, 5)
    assert result.top_periods
