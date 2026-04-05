from __future__ import annotations

import logging

import pytest
import torch

import altqft.nn.train as train
from altqft.nn.model import shift_ce_mean_loss_from_distributions


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

    def export_phases(self) -> list[float]:
        return self.phases.detach().cpu().tolist()


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
        train_device="cpu",
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
        train_device="cpu",
    )

    model = train.initialize_model(config, logging.getLogger("altqft.nn.train.test"))

    assert model.phases.detach().cpu().tolist() == pytest.approx([0.20, 0.80])


def test_run_training_tracks_best_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter((0.10, 0.35, 0.20))

    def fake_train_step(
        model: DummyModel,
        optimizer: torch.optim.Optimizer,
        config: train.TrainConfig,
    ) -> tuple[float, float | None, float | None]:
        del optimizer, config
        min_fi_value = next(values)
        with torch.no_grad():
            model.phases.copy_(torch.tensor([min_fi_value, -min_fi_value]))
        return -min_fi_value, min_fi_value, None

    monkeypatch.setattr(train, "train_step", fake_train_step)

    config = train.TrainConfig(
        nqubit=4,
        period_range=[2, 3],
        epochs=3,
        log_interval=10,
        train_device="cpu",
    )
    model = DummyModel(config.nqubit)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    history, best_checkpoint = train.run_training(
        model,
        optimizer,
        config,
        logging.getLogger("altqft.nn.train.test"),
    )

    assert [item.min_fi for item in history] == pytest.approx([0.10, 0.35, 0.20])
    assert best_checkpoint.epoch == 2
    assert best_checkpoint.min_fi == pytest.approx(0.35)
    assert best_checkpoint.phases == pytest.approx([0.35, -0.35])


def test_shift_ce_mean_loss_penalizes_mismatched_shift_distributions() -> None:
    identical = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mismatched = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    identical_loss, identical_l1 = shift_ce_mean_loss_from_distributions(identical)
    mismatched_loss, mismatched_l1 = shift_ce_mean_loss_from_distributions(mismatched)

    assert mismatched_loss > identical_loss
    assert identical_l1 == pytest.approx(0.0)
    assert mismatched_l1 > 0.0
