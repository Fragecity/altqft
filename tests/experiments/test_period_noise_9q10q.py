from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import altqft.experiments.period_noise_9q10q as period_noise_module
from altqft.experiments.period_noise_9q10q import (
    DEFAULT_EXPERIMENT_ROOTS,
    ExperimentRoots,
    PeriodNoiseRecipe,
    QubitExperimentSpec,
    apply_global_depolarizing_noise,
    legacy_qubit_experiment_spec,
    run_period_noise_experiment,
)


def test_apply_global_depolarizing_noise_returns_ideal_distribution_at_zero() -> None:
    probabilities = np.asarray([0.55, 0.35, 0.10], dtype=np.float64)

    noisy = apply_global_depolarizing_noise(probabilities, 0.0)

    assert np.allclose(noisy, probabilities)


def test_apply_global_depolarizing_noise_returns_uniform_distribution_at_one() -> None:
    probabilities = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)

    noisy = apply_global_depolarizing_noise(probabilities, 1.0)

    assert np.allclose(noisy, np.full(3, 1.0 / 3.0))


def test_apply_global_depolarizing_noise_stays_normalized_and_nonnegative() -> None:
    probabilities = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)

    noisy = apply_global_depolarizing_noise(probabilities, 0.125)

    assert np.isclose(noisy.sum(), 1.0)
    assert np.all(noisy >= 0.0)


def test_legacy_qubit_experiment_spec_matches_expected_ranges() -> None:
    spec_9 = legacy_qubit_experiment_spec(9)
    spec_10 = legacy_qubit_experiment_spec(10)

    assert spec_9.measurement_count == 82_944
    assert spec_9.period_min == 9
    assert spec_9.period_max == 80
    assert spec_10.measurement_count == 102_400
    assert spec_10.period_min == 10
    assert spec_10.period_max == 99


def test_default_experiment_roots_are_isolated_from_existing_outputs() -> None:
    assert DEFAULT_EXPERIMENT_ROOTS.model_dir == Path("model/noise_9q10q")
    assert DEFAULT_EXPERIMENT_ROOTS.output_dir == Path("outputs/noise_9q10q")
    assert DEFAULT_EXPERIMENT_ROOTS.dataset_dir == Path("data/period_recovery_noise_9q10q")


def test_run_period_noise_experiment_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALTQFT_TRAIN_DEVICE", "cpu")
    sample_call_count = 0
    original_sampler = period_noise_module._sample_bit_matrix_from_distribution

    def counting_sampler(
        distribution: np.ndarray,
        sample_count: int,
        basis_bit_rows: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        nonlocal sample_call_count
        sample_call_count += 1
        return original_sampler(distribution, sample_count, basis_bit_rows, rng)

    monkeypatch.setattr(
        period_noise_module,
        "_sample_bit_matrix_from_distribution",
        counting_sampler,
    )

    roots = ExperimentRoots(
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "outputs",
        dataset_dir=tmp_path / "datasets",
        data_dir=tmp_path / "data",
    )
    recipe = PeriodNoiseRecipe(
        variant_tag="noise-smoke",
        ph1_epochs=1,
        ph1_learning_rate=0.05,
        ph1_log_interval=1,
        ph1_train_device="cpu",
        top_k=2,
        batch_size=2,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        label_smoothing=0.0,
        min_epochs=1,
        early_stopping_patience=1,
        seed=13,
        log_interval=1,
        pool_multiplier=2,
        held_out_shifts_per_period=2,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=6,
        cache_device="cpu",
    )
    spec = QubitExperimentSpec(
        nqubit=4,
        measurement_count=16,
        period_min=4,
        period_max=5,
    )

    summary = run_period_noise_experiment(
        qubit_specs=(spec,),
        roots=roots,
        recipe=recipe,
        noise_levels=(1.0, 0.001),
    )

    assert summary.json_path.exists()
    assert summary.csv_path.exists()
    assert summary.png_path.exists()
    assert summary.svg_path.exists()
    assert summary.log_path.exists()
    assert len(summary.results) == 1
    assert summary.results[0].nqubit == 4
    assert len(summary.results[0].points) == 2
    assert summary.results[0].points[0].total == 8
    assert summary.results[0].points[1].total == 8
    assert summary.results[0].points[0].accuracy < 1.0
    assert sample_call_count == 16
    assert summary.results[0].model_path.parent == roots.model_dir
    assert summary.results[0].manifest_path.parent.parent == roots.dataset_dir
    payload = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["nqubit"] == 4
    assert len(payload["results"][0]["points"]) == 2
