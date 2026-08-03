from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from altqft.circuits.HPgenerators import HP1_parametrized
from altqft.nn.optimized_ph1 import OptimizedPH1Artifact, phase_artifact_is_current
from altqft.nn.period_decoder import (
    DeepSetPeriodPredictor,
    TOKEN_CLASSES,
    period_token_loss,
    predictor_from_checkpoint,
    periods_to_tokens,
)
from altqft.nn.period_recovery import (
    PoolBackedPeriodDataset,
    PeriodRecoveryDatasetConfig,
    PeriodRecoveryTrainConfig,
    generate_period_recovery_dataset,
    load_cached_dataset,
    load_shift_pool_manifest,
    summarize_bitmatrix_dataset,
    train_period_recovery,
)
from altqft.nn.periods import build_default_period_range


def build_optimized_artifact(
    tmp_path: Path,
    *,
    exact_support: bool = False,
    variant_tag: str | None = None,
    period_range: list[int] | None = None,
) -> OptimizedPH1Artifact:
    nqubit = 4
    resolved_period_range = period_range if period_range is not None else build_default_period_range(nqubit)
    phases = [0.1, 0.2, 0.3, 0.4]
    return OptimizedPH1Artifact(
        nqubit=nqubit,
        period_range=resolved_period_range,
        phases=phases,
        circuit=HP1_parametrized(nqubit, phases),
        objective="min_fi",
        exact_support=exact_support,
        variant_tag=variant_tag,
        model_path=tmp_path / "model.pt",
        phase_path=tmp_path / "phases.json",
        history_path=tmp_path / "history.json",
        log_path=tmp_path / "train.log",
        reused_existing=True,
        final_min_fi=None,
        final_loss=None,
        final_mean_shift_l1=None,
    )


def test_phase_artifact_is_current_detects_old_period_range() -> None:
    current_range = build_default_period_range(4)
    current_payload = {
        "nqubit": 4,
        "period_range": current_range,
        "phases": [0.1, 0.2, 0.3, 0.4],
        "objective": "min_fi",
        "exact_support": False,
        "variant_tag": None,
    }
    stale_payload = {
        "nqubit": 4,
        "period_range": [2, 3, 4, 5, 6, 7],
        "phases": [0.1, 0.2, 0.3, 0.4],
        "objective": "min_fi",
        "exact_support": False,
        "variant_tag": None,
    }

    assert phase_artifact_is_current(
        current_payload,
        4,
        current_range,
        objective="min_fi",
        exact_support=False,
        variant_tag=None,
    )
    assert not phase_artifact_is_current(
        stale_payload,
        4,
        current_range,
        objective="min_fi",
        exact_support=False,
        variant_tag=None,
    )


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
    assert torch.equal(cached_train.labels, cached_train.periods)
    assert set(cached_train.labels.tolist()).issubset(
        set(cached_train.candidate_periods.tolist())
    )
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
    checkpoint = torch.load(artifacts.model_path, map_location="cpu", weights_only=False)
    assert checkpoint["decoder_type"] == "nibble"
    assert checkpoint["token_bits"] == 4
    assert checkpoint["beam_width"] == 4
    assert checkpoint["state_dict"]["token_classifier.weight"].shape[0] == 16
    assert "classifier.weight" not in checkpoint["state_dict"]
    loaded_model, loaded_periods = predictor_from_checkpoint(checkpoint, nqubit=4)
    assert loaded_model.token_count == 1
    assert loaded_periods == tuple(build_default_period_range(4))


def test_checkpoint_loader_rejects_removed_class_decoder() -> None:
    with pytest.raises(ValueError, match="decoder_type"):
        predictor_from_checkpoint(
            {
                "candidate_periods": [2, 3],
                "state_dict": {},
                "decoder_type": "class",
                "token_bits": 4,
                "model_architecture": "weighted",
            },
            nqubit=2,
        )


def test_generate_shift_pool_cache_writes_manifest_and_shards(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(
        tmp_path,
        exact_support=True,
        variant_tag="shiftpool-smoke",
        period_range=[4, 5, 6],
    )
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=16,
        period_min=4,
        period_max=6,
        seed=17,
        dataset_dir=tmp_path / "datasets",
        exact_support=True,
        cache_mode="shift_pool",
        pool_multiplier=2,
        held_out_shifts_per_period=1,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=8,
        variant_tag="shiftpool-smoke",
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    manifest = load_shift_pool_manifest(dataset_artifacts.manifest_path or dataset_artifacts.train_path)

    assert dataset_artifacts.cache_mode == "shift_pool"
    assert manifest.measurement_count == 16
    assert manifest.pool_size == 32
    assert manifest.train_draws_per_epoch == 8
    assert manifest.candidate_periods == (4, 5, 6)
    assert len(manifest.shards) == 3
    for shard in manifest.shards:
        assert len(shard.val_shifts) == 1
        assert set(shard.train_shifts).isdisjoint(set(shard.val_shifts))
        assert set(shard.train_shifts) | set(shard.val_shifts) == set(range(shard.period))
        shard_path = manifest.manifest_path.parent / shard.path
        assert shard_path.exists()


def test_pool_backed_dataset_returns_expected_shapes_and_metadata(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path, exact_support=True, period_range=[4, 5])
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=16,
        period_min=4,
        period_max=5,
        seed=19,
        dataset_dir=tmp_path / "datasets",
        exact_support=True,
        cache_mode="shift_pool",
        pool_multiplier=2,
        held_out_shifts_per_period=1,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=6,
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    manifest = load_shift_pool_manifest(dataset_artifacts.manifest_path or dataset_artifacts.train_path)
    train_dataset = PoolBackedPeriodDataset(manifest, split="train")
    val_dataset = PoolBackedPeriodDataset(manifest, split="val")
    train_dataset.set_epoch(0)
    sample_bits, sample_label = train_dataset[0]
    val_entry = val_dataset.describe_index(0)

    assert sample_bits.shape == (16, 4)
    assert sample_bits.dtype == torch.int8
    assert sample_label.dtype == torch.long
    assert sample_label.item() == train_dataset.describe_index(0).period
    assert len(train_dataset) == 6
    assert len(val_dataset) == len(manifest.candidate_periods) * 2
    assert 0 <= val_entry.shift < val_entry.period
    train_entries = [train_dataset.describe_index(index) for index in range(len(train_dataset))]
    assert train_entries != sorted(
        train_entries,
        key=lambda entry: (entry.period, entry.shift, entry.draw_index),
    )


def test_pool_backed_dataset_groups_random_epoch_entries_by_batch_size(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path, exact_support=True, period_range=[4, 5])
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=16,
        period_min=4,
        period_max=5,
        seed=29,
        dataset_dir=tmp_path / "datasets",
        exact_support=True,
        cache_mode="shift_pool",
        pool_multiplier=2,
        held_out_shifts_per_period=1,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=6,
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    manifest = load_shift_pool_manifest(dataset_artifacts.manifest_path or dataset_artifacts.train_path)
    train_dataset = PoolBackedPeriodDataset(manifest, split="train")
    train_dataset.set_epoch(0, batch_size=2)
    train_entries = [train_dataset.describe_index(index) for index in range(len(train_dataset))]

    assert train_entries != sorted(
        train_entries,
        key=lambda entry: (entry.period, entry.shift, entry.draw_index),
    )
    for start in range(0, len(train_entries), 2):
        periods = {entry.period for entry in train_entries[start : start + 2]}
        assert len(periods) == 1


def test_pool_backed_dataset_changes_batch_order_between_epochs(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path, exact_support=True, period_range=[4, 5, 6])
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=16,
        period_min=4,
        period_max=6,
        seed=31,
        dataset_dir=tmp_path / "datasets",
        exact_support=True,
        cache_mode="shift_pool",
        pool_multiplier=2,
        held_out_shifts_per_period=1,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=12,
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    manifest = load_shift_pool_manifest(dataset_artifacts.manifest_path or dataset_artifacts.train_path)
    train_dataset = PoolBackedPeriodDataset(manifest, split="train")

    train_dataset.set_epoch(0, batch_size=2)
    first_epoch = [train_dataset.describe_index(index) for index in range(len(train_dataset))]
    train_dataset.set_epoch(0, batch_size=2)
    repeated_first_epoch = [train_dataset.describe_index(index) for index in range(len(train_dataset))]
    train_dataset.set_epoch(1, batch_size=2)
    second_epoch = [train_dataset.describe_index(index) for index in range(len(train_dataset))]

    assert first_epoch == repeated_first_epoch
    assert first_epoch != second_epoch
    assert first_epoch != sorted(first_epoch, key=lambda entry: (entry.period, entry.shift, entry.draw_index))
    assert second_epoch != sorted(second_epoch, key=lambda entry: (entry.period, entry.shift, entry.draw_index))


def test_deepset_accepts_measurement_prefixes_from_cached_dataset(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(tmp_path)
    config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=16,
        num_train_samples=6,
        num_val_samples=2,
        seed=37,
        dataset_dir=tmp_path / "datasets",
    )

    dataset_artifacts = generate_period_recovery_dataset(config, artifact, regenerate=True)
    cached_train = load_cached_dataset(dataset_artifacts.train_path)
    candidate_periods = cached_train.candidate_periods.tolist()
    model = DeepSetPeriodPredictor(
        nqubit=4,
        num_periods=len(candidate_periods),
        period_min=candidate_periods[0],
        period_max=candidate_periods[-1],
    )
    inputs = cached_train.bit_matrices[:3]

    for measurement_count in (4, 8, 16):
        logits = model(inputs[:, :measurement_count, :])
        assert logits.shape == (3, model.token_count, TOKEN_CLASSES)
        assert torch.isfinite(logits).any(dim=-1).all()


def test_train_period_recovery_runs_with_shift_pool_dataset(tmp_path: Path) -> None:
    artifact = build_optimized_artifact(
        tmp_path,
        exact_support=True,
        variant_tag="shiftpool-train",
        period_range=[4, 5, 6],
    )
    dataset_config = PeriodRecoveryDatasetConfig(
        nqubit=4,
        measurement_count=32,
        period_min=4,
        period_max=6,
        seed=23,
        dataset_dir=tmp_path / "datasets",
        exact_support=True,
        cache_mode="shift_pool",
        pool_multiplier=2,
        held_out_shifts_per_period=1,
        val_draws_per_heldout_shift=2,
        train_draws_per_epoch=12,
        variant_tag="shiftpool-train",
    )
    train_config = PeriodRecoveryTrainConfig(
        nqubit=4,
        period_min=4,
        period_max=6,
        top_k=2,
        batch_size=4,
        epochs=3,
        min_epochs=2,
        early_stopping_patience=2,
        seed=23,
        log_interval=1,
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        regenerate_dataset=True,
        fi_epochs=1,
        dataset_mode="shift_pool",
        variant_tag="shiftpool-train",
    )

    dataset_artifacts = generate_period_recovery_dataset(dataset_config, artifact, regenerate=True)
    artifacts = train_period_recovery(train_config, dataset_artifacts, artifact)

    assert artifacts.model_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.selected_epoch >= 1


def test_deepset_model_returns_four_bit_token_logits() -> None:
    candidate_periods = build_default_period_range(4)
    model = DeepSetPeriodPredictor(
        nqubit=4,
        num_periods=len(candidate_periods),
        period_min=candidate_periods[0],
        period_max=candidate_periods[-1],
    )
    inputs = torch.randint(0, 2, (3, 8, 4), dtype=torch.int8)
    periods = torch.tensor([2, 7, 15])
    token_logits = model(inputs, target_periods=periods)
    loss = period_token_loss(token_logits, periods)

    assert model.bit_width == 4
    assert model.beam_width == 4
    assert token_logits.shape == (3, 1, 16)
    assert model.token_classifier.out_features == 16
    assert not hasattr(model, "classifier")
    assert torch.isfinite(loss)


def test_periods_are_encoded_as_direct_four_bit_tokens() -> None:
    periods = torch.tensor([2, 15, 16, 255])

    tokens = periods_to_tokens(periods, token_count=2)

    assert tokens.tolist() == [[0, 2], [0, 15], [1, 0], [15, 15]]


def test_nibble_decoder_parameter_count_does_not_depend_on_period_count() -> None:
    small_range_model = DeepSetPeriodPredictor(
        nqubit=8,
        num_periods=14,
        period_min=2,
        period_max=15,
    )
    large_range_model = DeepSetPeriodPredictor(
        nqubit=8,
        num_periods=254,
        period_min=2,
        period_max=255,
    )

    small_parameter_count = sum(
        parameter.numel() for parameter in small_range_model.parameters()
    )
    large_parameter_count = sum(
        parameter.numel() for parameter in large_range_model.parameters()
    )

    assert small_parameter_count == large_parameter_count


def test_nibble_beam_keeps_four_period_prefixes() -> None:
    model = DeepSetPeriodPredictor(
        nqubit=8,
        num_periods=254,
        period_min=2,
        period_max=255,
        beam_width=4,
    )
    inputs = torch.randint(0, 2, (2, 8, 8), dtype=torch.int8)

    periods, bits, scores = model.predict_topk_periods(
        inputs,
        range(2, 256),
        k=4,
    )

    assert periods.shape == (2, 4)
    assert bits.shape == (2, 4, 8)
    assert scores.shape == (2, 4)
    assert ((2 <= periods) & (periods <= 255)).all()
