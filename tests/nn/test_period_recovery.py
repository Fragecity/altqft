from __future__ import annotations

import math
from pathlib import Path

import torch

from altqft.circuits.ph_generators import ph_1_parametrized
from altqft.nn.optimized_ph1 import OptimizedPH1Artifact, phase_artifact_is_current
from altqft.nn.period_recovery import (
    DeepSetPeriodPredictor,
    PeriodRecoveryDatasetConfig,
    PeriodRecoveryTrainConfig,
    compact_label_bit_width,
    decode_topk_periods,
    generate_period_recovery_dataset,
    load_cached_dataset,
    period_class_loss,
    period_bit_loss,
    summarize_bitmatrix_dataset,
    topk_accuracy,
    train_period_recovery,
)
from altqft.nn.periods import build_default_period_range


def build_optimized_artifact(tmp_path: Path) -> OptimizedPH1Artifact:
    nqubit = 4
    period_range = build_default_period_range(nqubit)
    phases = [0.1, 0.2, 0.3, 0.4]
    return OptimizedPH1Artifact(
        nqubit=nqubit,
        period_range=period_range,
        phases=phases,
        circuit=ph_1_parametrized(nqubit, phases),
        model_path=tmp_path / "model.pt",
        phase_path=tmp_path / "phases.json",
        history_path=tmp_path / "history.json",
        log_path=tmp_path / "train.log",
        reused_existing=True,
        final_min_fi=None,
    )


def test_phase_artifact_is_current_detects_old_period_range() -> None:
    current_range = build_default_period_range(4)
    current_payload = {
        "nqubit": 4,
        "period_range": current_range,
        "phases": [0.1, 0.2, 0.3, 0.4],
    }
    stale_payload = {
        "nqubit": 4,
        "period_range": [2, 3, 4, 5, 6, 7],
        "phases": [0.1, 0.2, 0.3, 0.4],
    }

    assert phase_artifact_is_current(current_payload, 4, current_range)
    assert not phase_artifact_is_current(stale_payload, 4, current_range)


def test_generate_period_recovery_dataset_caches_expected_tensors(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path)
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=32,
        num_train_samples=4,
        num_val_samples=2,
        seed=11,
        dataset_dir=tmp_path / "datasets",
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    cached_train = load_cached_dataset(dataset_artifacts.train_path)
    cached_val = load_cached_dataset(dataset_artifacts.val_path)

    assert cached_train.bit_matrices.shape == (4, 32, 4)
    assert cached_train.bit_matrices.dtype == torch.int8
    assert cached_train.labels.shape == (4,)
    assert cached_train.periods.shape == (4,)
    assert cached_train.shifts.shape == (4,)
    assert cached_train.candidate_periods.tolist() == build_default_period_range(4)
    assert cached_train.labels.min().item() >= 0
    assert cached_train.labels.max().item() < len(cached_train.candidate_periods)
    assert cached_val.bit_matrices.shape == (2, 32, 4)
    assert set(cached_val.periods.tolist()).issubset(set(cached_train.periods.tolist()))


def test_summarize_bitmatrix_dataset_reports_sample_class_ratio(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path)
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=32,
        num_train_samples=4,
        num_val_samples=2,
        seed=11,
        dataset_dir=tmp_path / "datasets",
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    cached_train = load_cached_dataset(dataset_artifacts.train_path)
    summary = summarize_bitmatrix_dataset(cached_train)

    assert summary.sample_count == 4
    assert summary.measurement_count == 32
    assert summary.state_space_size == 16
    assert math.isclose(summary.measurements_per_basis_state, 2.0, rel_tol=1e-6)
    assert math.isclose(
        summary.samples_per_class,
        4 / len(build_default_period_range(4)),
        rel_tol=1e-6,
    )


def test_train_period_recovery_runs_end_to_end_for_4q(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path)
    dataset_config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=128,
        num_train_samples=8,
        num_val_samples=4,
        seed=13,
        dataset_dir=tmp_path / "datasets",
    )
    train_config = PeriodRecoveryTrainConfig(
        nqubit=4,
        top_k=2,
        batch_size=4,
        epochs=4,
        min_epochs=2,
        early_stopping_patience=2,
        seed=13,
        log_interval=1,
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        force_reoptimize_phases=False,
        regenerate_dataset=True,
        fi_epochs=1,
    )

    dataset_artifacts = generate_period_recovery_dataset(
        dataset_config,
        artifact,
        regenerate=True,
    )
    artifacts = train_period_recovery(train_config, dataset_artifacts, artifact)

    assert artifacts.model_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.log_path.exists()
    assert artifacts.selected_epoch >= 1
    assert artifacts.last_epoch == len(artifacts.history)
    assert artifacts.last_epoch <= train_config.epochs
    assert 0.0 <= artifacts.selected_val_top1 <= 1.0
    assert 0.0 <= artifacts.selected_val_topk <= 1.0


def test_deepset_model_returns_expected_class_logits_shape() -> None:
    num_periods = len(build_default_period_range(4))
    model = DeepSetPeriodPredictor(nqubit=4, num_periods=num_periods)
    inputs = torch.randint(0, 2, (3, 8, 4), dtype=torch.int8)
    class_logits = model(inputs)
    loss = period_class_loss(class_logits, torch.tensor([0, 1, 2]))

    assert model.bit_width == compact_label_bit_width(num_periods)
    assert class_logits.shape == (3, num_periods)
    assert torch.isfinite(loss)


def test_period_bit_loss_accepts_bitwise_logits() -> None:
    bit_logits = torch.tensor(
        [
            [[5.0, 0.0], [4.0, 0.0]],
            [[4.0, 2.0], [3.0, 2.0]],
            [[4.0, 2.0], [4.0, 3.0]],
        ]
    )

    loss = period_bit_loss(bit_logits, torch.tensor([0, 1, 2]))

    assert torch.isfinite(loss)


def test_topk_accuracy_counts_hits_in_top_k() -> None:
    bit_logits = torch.tensor(
        [
            [[5.0, 0.0], [4.0, 0.0]],
            [[4.0, 2.0], [3.0, 2.0]],
            [[4.0, 2.0], [4.0, 3.0]],
        ]
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    assert math.isclose(topk_accuracy(bit_logits, labels, 1, num_classes=4), 1 / 3, rel_tol=1e-6)
    assert math.isclose(topk_accuracy(bit_logits, labels, 2, num_classes=4), 2 / 3, rel_tol=1e-6)


def test_topk_accuracy_supports_class_logits() -> None:
    class_logits = torch.tensor(
        [
            [5.0, 0.0, -1.0, -2.0],
            [0.0, 4.0, 3.0, -1.0],
            [0.0, -1.0, 5.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 1, 3], dtype=torch.long)

    assert math.isclose(topk_accuracy(class_logits, labels, 1, num_classes=4), 2 / 3, rel_tol=1e-6)
    assert math.isclose(topk_accuracy(class_logits, labels, 2, num_classes=4), 1.0, rel_tol=1e-6)


def test_decode_topk_periods_prunes_invalid_bit_patterns() -> None:
    candidate_periods = [4, 5, 6, 7, 8]
    bit_logits = torch.tensor(
        [
            [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
        ]
    )

    top_periods, top_bits, top_scores = decode_topk_periods(bit_logits, candidate_periods, k=3)

    assert top_periods.shape == (1, 3)
    assert top_bits.shape == (1, 3, compact_label_bit_width(len(candidate_periods)))
    assert top_scores.shape == (1, 3)
    assert top_periods[0, 0].item() == 7
    assert top_bits[0, 0].tolist() == [0, 1, 1]
    assert set(top_periods[0].tolist()).issubset(candidate_periods)
