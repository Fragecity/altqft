from __future__ import annotations

from pathlib import Path

import pytest
import torch
from qiskit import QuantumCircuit

import altqft.algorithms.shor_ph1 as shor_ph1
from altqft.algorithms.shor_ph1 import (
    PH1ShorConfig,
    _collapsed_periodic_state,
    _modular_exponentiation_outputs,
    run_shor_with_ph1,
)
from altqft.nn.optimized_ph1 import OptimizedPH1Artifact


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


class FakePeriodModel:
    def predict_topk_periods(self, bit_matrices, candidate_periods, k):
        del bit_matrices, candidate_periods, k
        return (
            torch.tensor([[10, 12, 4]], dtype=torch.long),
            torch.empty((1, 3, 1), dtype=torch.long),
            torch.tensor([[0.9, 0.8, 0.1]], dtype=torch.float32),
        )


def test_run_shor_with_ph1_factors_15_with_repo_artifacts(monkeypatch) -> None:
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
    candidate_periods = list(config.candidate_periods)

    def fake_ensure_optimized_ph1(*args, **kwargs):
        del args, kwargs
        return OptimizedPH1Artifact(
            nqubit=config.nqubit,
            period_range=candidate_periods,
            phases=[],
            circuit=QuantumCircuit(config.nqubit),
            objective="min_fi",
            exact_support=False,
            variant_tag=None,
            model_path=Path("model/ph1_min_fi_4q.pt"),
            phase_path=Path("model/ph1_min_fi_4q_phases.json"),
            history_path=Path("outputs/ph1_min_fi_4q_history.json"),
            log_path=Path("outputs/ph1_min_fi_4q.log"),
            reused_existing=True,
            final_min_fi=None,
            final_loss=None,
            final_mean_shift_l1=None,
        )

    def fake_load_period_recovery_model(loaded_config):
        assert loaded_config is config
        return FakePeriodModel(), tuple(candidate_periods), Path("model/period_recovery_4q.pt")

    monkeypatch.setattr(shor_ph1, "ensure_optimized_ph1", fake_ensure_optimized_ph1)
    monkeypatch.setattr(
        shor_ph1,
        "_load_period_recovery_model",
        fake_load_period_recovery_model,
    )

    result = run_shor_with_ph1(config)

    assert result.success
    assert result.top1_period == 10
    assert result.predicted_period == 12
    assert result.factors == (3, 5)
    assert result.top_periods
