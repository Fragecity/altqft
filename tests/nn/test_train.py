from __future__ import annotations

import logging

import pytest
import torch

import altqft.nn.train as train


class DummyModel(torch.nn.Module):
    def __init__(self, nqubit: int, init_phases: list[float] | None = None) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.phase_count = 2
        phases = init_phases if init_phases is not None else [0.0, 0.0]
        self.phases = torch.nn.Parameter(torch.tensor(phases, dtype=torch.float32))

    def forward(self, period_range: list[int]) -> torch.Tensor:
        del period_range
        target = torch.tensor([0.75, 0.25], dtype=self.phases.dtype)
        return -torch.square(self.phases - target).sum()


def test_select_monte_carlo_init_phases_picks_best_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = iter(
        (
            torch.tensor([0.10, 0.10], dtype=torch.float32),
            torch.tensor([0.75, 0.25], dtype=torch.float32),
            torch.tensor([0.40, 0.30], dtype=torch.float32),
        )
    )

    def fake_sample_phase_tensor(phase_count: int) -> torch.Tensor:
        sample = next(samples)
        assert sample.numel() == phase_count
        return sample

    monkeypatch.setattr(train, "PH1MinFIModel", DummyModel)
    monkeypatch.setattr(train, "sample_phase_tensor", fake_sample_phase_tensor)

    config = train.TrainConfig(
        nqubit=4,
        period_range=[2, 3],
        monte_carlo_samples=3,
    )

    init_phases, init_min_fi = train.select_monte_carlo_init_phases(config)

    assert init_phases == pytest.approx([0.75, 0.25])
    assert init_min_fi == pytest.approx(0.0)


def test_initialize_model_uses_selected_monte_carlo_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "PH1MinFIModel", DummyModel)
    monkeypatch.setattr(
        train,
        "select_monte_carlo_init_phases",
        lambda config: ([0.20, 0.80], 1.23),
    )

    config = train.TrainConfig(
        nqubit=4,
        period_range=[2, 3],
        monte_carlo_samples=5,
    )

    model = train.initialize_model(config, logging.getLogger("altqft.nn.train.test"))

    assert model.phases.detach().cpu().tolist() == pytest.approx([0.20, 0.80])
