from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from altqft.nn.model import OBJECTIVES
from altqft.nn.optimized_ph1 import OptimizedPH1Artifact
from altqft.nn.period_decoder import (
    DECODER_TYPE,
    DEFAULT_BEAM_WIDTH,
    TOKEN_BITS,
    DeepSetPeriodPredictor,
    period_token_loss,
)
from altqft.nn.periods import build_period_range, period_range_artifact_suffix
from altqft.nn.devices import resolve_compute_device
from altqft.nn.process_qc import (
    ProbFunc,
    _torch_exact_support_indices,
    _torch_probability_vector_from_support,
    _torch_surrogate_support_indices,
    _torch_unitary,
    make_prob,
    probability_distribution,
)
from altqft.nn.runtime import configure_logger, set_random_seed, snapshot_model_state

LOGGER_NAME = "altqft.nn.period_recovery"
DatasetPayload = dict[str, Tensor | dict[str, Any]]
TrainBatch = tuple[Tensor, Tensor]
DATASET_MODES = {"flat", "shift_pool"}


@dataclass(slots=True)
class PeriodRecoveryDatasetConfig:
    nqubit: int
    measurement_count: int
    num_train_samples: int = 0
    num_val_samples: int = 0
    period_min: int = 2
    period_max: int | None = None
    seed: int = 7
    stratify_periods: bool = True
    dataset_dir: Path = Path("data/period_recovery")
    exact_support: bool = False
    cache_mode: str = "flat"
    pool_multiplier: int = 1
    held_out_shifts_per_period: int = 1
    val_draws_per_heldout_shift: int = 4
    train_draws_per_epoch: int | None = None
    variant_tag: str | None = None
    cache_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if self.measurement_count < 1:
            raise ValueError("measurement_count must be positive")
        if self.cache_mode not in DATASET_MODES:
            supported = ", ".join(sorted(DATASET_MODES))
            raise ValueError(f"unsupported cache_mode '{self.cache_mode}', expected one of: {supported}")
        if self.cache_mode == "flat":
            if self.num_train_samples < 1:
                raise ValueError("num_train_samples must be positive")
            if self.num_val_samples < 1:
                raise ValueError("num_val_samples must be positive")
        else:
            if self.pool_multiplier < 1:
                raise ValueError("pool_multiplier must be positive")
            if self.held_out_shifts_per_period < 1:
                raise ValueError("held_out_shifts_per_period must be positive")
            if self.val_draws_per_heldout_shift < 1:
                raise ValueError("val_draws_per_heldout_shift must be positive")
            if self.train_draws_per_epoch is not None and self.train_draws_per_epoch < 1:
                raise ValueError("train_draws_per_epoch must be positive when provided")
            for period in self.candidate_periods:
                if self.held_out_shifts_per_period >= period:
                    raise ValueError("held_out_shifts_per_period must be smaller than every candidate period")

    @property
    def candidate_periods(self) -> list[int]:
        return build_period_range(
            self.nqubit,
            min_period=self.period_min,
            max_period=self.period_max,
        )

    @property
    def dataset_stem(self) -> str:
        suffix = period_range_artifact_suffix(self.nqubit, self.candidate_periods)
        base_stem = f"period_recovery_{self.nqubit}q{suffix}"
        if self.variant_tag:
            base_stem = f"{base_stem}_{self.variant_tag}"
        if self.cache_mode == "flat":
            return (
                f"{base_stem}_"
                f"m{self.measurement_count}_"
                f"train{self.num_train_samples}_"
                f"val{self.num_val_samples}_"
                f"seed{self.seed}"
            )
        return f"{base_stem}_m{self.measurement_count}_seed{self.seed}"

    @property
    def pool_size(self) -> int:
        return self.measurement_count * self.pool_multiplier

    @property
    def resolved_train_draws_per_epoch(self) -> int:
        if self.cache_mode != "shift_pool":
            return self.num_train_samples
        if self.train_draws_per_epoch is not None:
            return self.train_draws_per_epoch
        return sum(period - self.held_out_shifts_per_period for period in self.candidate_periods)

    @property
    def train_path(self) -> Path:
        if self.cache_mode == "shift_pool":
            return self.manifest_path
        return self.dataset_dir / f"{self.dataset_stem}_train.pt"

    @property
    def val_path(self) -> Path:
        if self.cache_mode == "shift_pool":
            return self.manifest_path
        return self.dataset_dir / f"{self.dataset_stem}_val.pt"

    @property
    def cache_root(self) -> Path:
        return self.dataset_dir / self.dataset_stem

    @property
    def manifest_path(self) -> Path:
        return self.cache_root / "manifest.json"

    def period_shard_path(self, period: int) -> Path:
        return self.cache_root / f"period_{period:03d}.pt"


@dataclass(slots=True)
class PeriodRecoveryTrainConfig:
    nqubit: int
    period_min: int = 2
    period_max: int | None = None
    top_k: int = 3
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    label_smoothing: float = 0.05
    min_epochs: int = 25
    early_stopping_patience: int = 50
    seed: int = 7
    log_interval: int = 1
    model_dir: Path = Path("model")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    force_reoptimize_phases: bool = False
    regenerate_dataset: bool = False
    fi_epochs: int = 100
    fi_learning_rate: float = 0.05
    fi_log_interval: int = 10
    fi_objective: str = "min_fi"
    fi_exact_support: bool = False
    fi_train_device: str = "auto"
    dataset_mode: str = "flat"
    variant_tag: str | None = None

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if not 1 <= self.top_k <= DEFAULT_BEAM_WIDTH:
            raise ValueError(
                f"top_k must be between 1 and beam width {DEFAULT_BEAM_WIDTH}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
        if self.fi_epochs < 1:
            raise ValueError("fi_epochs must be positive")
        if self.fi_log_interval < 1:
            raise ValueError("fi_log_interval must be positive")
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs must not exceed epochs")
        if self.dataset_mode not in DATASET_MODES:
            supported = ", ".join(sorted(DATASET_MODES))
            raise ValueError(f"unsupported dataset_mode '{self.dataset_mode}', expected one of: {supported}")
        if self.fi_objective not in OBJECTIVES:
            supported = ", ".join(sorted(OBJECTIVES))
            raise ValueError(f"unsupported fi_objective '{self.fi_objective}', expected one of: {supported}")

    @property
    def run_name(self) -> str:
        suffix = period_range_artifact_suffix(self.nqubit, self.candidate_periods)
        base_name = f"period_recovery_{self.nqubit}q{suffix}"
        if self.variant_tag:
            return f"{base_name}_{self.variant_tag}"
        return base_name

    @property
    def candidate_periods(self) -> list[int]:
        return build_period_range(
            self.nqubit,
            min_period=self.period_min,
            max_period=self.period_max,
        )

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.run_name}.pt"

    @property
    def history_path(self) -> Path:
        return self.output_dir / f"{self.run_name}_history.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.log"


@dataclass(frozen=True, slots=True)
class CachedPeriodDataset:
    bit_matrices: Tensor
    labels: Tensor
    periods: Tensor
    shifts: Tensor
    candidate_periods: Tensor
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PeriodRecoveryDatasetArtifacts:
    candidate_periods: list[int]
    train_path: Path
    val_path: Path
    cache_mode: str = "flat"
    manifest_path: Path | None = None


@dataclass(slots=True)
class PeriodRecoveryEpochResult:
    epoch: int
    train_loss: float
    train_top1: float
    train_topk: float
    val_loss: float
    val_top1: float
    val_topk: float


@dataclass(frozen=True, slots=True)
class PeriodRecoveryDataDiagnostics:
    sample_count: int
    measurement_count: int
    nqubit: int
    state_space_size: int
    candidate_period_count: int
    measurements_per_basis_state: float
    samples_per_class: float
    unique_period_count: int | None = None


@dataclass(frozen=True, slots=True)
class PeriodRecoveryCheckpoint:
    result: PeriodRecoveryEpochResult
    state_dict: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class PeriodRecoveryTrainArtifacts:
    history: list[PeriodRecoveryEpochResult]
    top_k: int
    selected_epoch: int
    selected_train_top1: float
    selected_train_topk: float
    selected_val_top1: float
    selected_val_topk: float
    last_epoch: int
    last_train_top1: float
    last_train_topk: float
    last_val_top1: float
    last_val_topk: float
    stopped_early: bool
    model_path: Path
    history_path: Path
    log_path: Path
    train_dataset_path: Path
    val_dataset_path: Path
    phase_path: Path


@dataclass(frozen=True, slots=True)
class ShiftPoolShardInfo:
    period: int
    label: int
    path: str
    train_shifts: tuple[int, ...]
    val_shifts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShiftPoolManifest:
    manifest_path: Path
    nqubit: int
    measurement_count: int
    pool_size: int
    pool_multiplier: int
    held_out_shifts_per_period: int
    val_draws_per_heldout_shift: int
    train_draws_per_epoch: int
    seed: int
    exact_support: bool
    variant_tag: str | None
    candidate_periods: tuple[int, ...]
    shards: tuple[ShiftPoolShardInfo, ...]
    config: dict[str, Any]

    @property
    def unique_period_count(self) -> int:
        return len(self.candidate_periods)


@dataclass(frozen=True, slots=True)
class PoolDrawEntry:
    period: int
    shift: int
    label: int
    draw_index: int


def serialize_dataset_config(
    config: PeriodRecoveryDatasetConfig,
    *,
    split: str,
) -> dict[str, int | str | bool | list[int] | None]:
    return {
        "nqubit": config.nqubit,
        "measurement_count": config.measurement_count,
        "num_train_samples": config.num_train_samples,
        "num_val_samples": config.num_val_samples,
        "period_min": config.period_min,
        "period_max": config.period_max,
        "seed": config.seed,
        "stratify_periods": config.stratify_periods,
        "dataset_dir": str(config.dataset_dir),
        "candidate_periods": config.candidate_periods,
        "label_encoding": "period",
        "exact_support": config.exact_support,
        "cache_mode": config.cache_mode,
        "pool_multiplier": config.pool_multiplier,
        "held_out_shifts_per_period": config.held_out_shifts_per_period,
        "val_draws_per_heldout_shift": config.val_draws_per_heldout_shift,
        "train_draws_per_epoch": config.train_draws_per_epoch,
        "variant_tag": config.variant_tag,
        "cache_device": config.cache_device,
        "split": split,
    }


def serialize_train_config(
    config: PeriodRecoveryTrainConfig,
) -> dict[str, int | float | str | bool | None]:
    return {
        "nqubit": config.nqubit,
        "period_min": config.period_min,
        "period_max": config.period_max,
        "top_k": config.top_k,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "label_smoothing": config.label_smoothing,
        "min_epochs": config.min_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "seed": config.seed,
        "log_interval": config.log_interval,
        "model_dir": str(config.model_dir),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "force_reoptimize_phases": config.force_reoptimize_phases,
        "regenerate_dataset": config.regenerate_dataset,
        "fi_epochs": config.fi_epochs,
        "fi_learning_rate": config.fi_learning_rate,
        "fi_log_interval": config.fi_log_interval,
        "fi_objective": config.fi_objective,
        "fi_exact_support": config.fi_exact_support,
        "fi_train_device": config.fi_train_device,
        "dataset_mode": config.dataset_mode,
        "variant_tag": config.variant_tag,
    }


def summarize_bitmatrix_dataset(
    dataset: CachedPeriodDataset | PoolBackedPeriodDataset,
    *,
    measurement_count: int | None = None,
) -> PeriodRecoveryDataDiagnostics:
    if isinstance(dataset, PoolBackedPeriodDataset):
        resolved_measurement_count = (
            int(measurement_count)
            if measurement_count is not None
            else int(dataset.measurement_count)
        )
        state_space_size = 1 << dataset.nqubit
        return PeriodRecoveryDataDiagnostics(
            sample_count=len(dataset),
            measurement_count=resolved_measurement_count,
            nqubit=dataset.nqubit,
            state_space_size=state_space_size,
            candidate_period_count=len(dataset.candidate_periods),
            measurements_per_basis_state=resolved_measurement_count / float(state_space_size),
            samples_per_class=len(dataset) / float(len(dataset.candidate_periods)),
            unique_period_count=len(dataset.candidate_periods),
        )

    sample_count = int(dataset.bit_matrices.shape[0])
    resolved_measurement_count = (
        int(measurement_count)
        if measurement_count is not None
        else int(dataset.bit_matrices.shape[1])
    )
    nqubit = int(dataset.bit_matrices.shape[2])
    candidate_period_count = int(dataset.candidate_periods.numel())
    unique_period_count = int(torch.unique(dataset.periods).numel())
    state_space_size = 1 << nqubit
    return PeriodRecoveryDataDiagnostics(
        sample_count=sample_count,
        measurement_count=resolved_measurement_count,
        nqubit=nqubit,
        state_space_size=state_space_size,
        candidate_period_count=candidate_period_count,
        measurements_per_basis_state=resolved_measurement_count / float(state_space_size),
        samples_per_class=sample_count / float(candidate_period_count),
        unique_period_count=unique_period_count,
    )


def _log_dataset_diagnostics(
    logger: logging.Logger,
    split: str,
    diagnostics: PeriodRecoveryDataDiagnostics,
) -> None:
    logger.info(
        (
            "%s dataset diagnostics samples=%s measurement_count=%s nqubit=%s "
            "state_space=%s candidate_periods=%s unique_periods=%s "
            "measurements_per_basis_state=%.2f samples_per_class=%.4f"
        ),
        split,
        diagnostics.sample_count,
        diagnostics.measurement_count,
        diagnostics.nqubit,
        diagnostics.state_space_size,
        diagnostics.candidate_period_count,
        diagnostics.unique_period_count,
        diagnostics.measurements_per_basis_state,
        diagnostics.samples_per_class,
    )


def _held_out_shifts_for_period(
    period: int,
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    if count >= period:
        raise ValueError("count must be smaller than period")
    rng = np.random.default_rng(seed + period * 10_007)
    selected = np.sort(rng.choice(period, size=count, replace=False))
    return tuple(int(value) for value in selected.tolist())


def _serialize_shift_pool_manifest(
    config: PeriodRecoveryDatasetConfig,
    shards: Sequence[ShiftPoolShardInfo],
) -> dict[str, Any]:
    return {
        "config": serialize_dataset_config(config, split="shift_pool_manifest"),
        "nqubit": config.nqubit,
        "measurement_count": config.measurement_count,
        "pool_size": config.pool_size,
        "pool_multiplier": config.pool_multiplier,
        "held_out_shifts_per_period": config.held_out_shifts_per_period,
        "val_draws_per_heldout_shift": config.val_draws_per_heldout_shift,
        "train_draws_per_epoch": config.resolved_train_draws_per_epoch,
        "seed": config.seed,
        "exact_support": config.exact_support,
        "variant_tag": config.variant_tag,
        "candidate_periods": config.candidate_periods,
        "shards": [
            {
                "period": shard.period,
                "label": shard.label,
                "path": shard.path,
                "train_shifts": list(shard.train_shifts),
                "val_shifts": list(shard.val_shifts),
            }
            for shard in shards
        ],
    }


def _parse_shift_pool_shard_info(item: object, manifest_path: Path) -> ShiftPoolShardInfo:
    if not isinstance(item, dict):
        raise ValueError(f"invalid shift-pool shard entry in {manifest_path}")
    period = item.get("period")
    label = item.get("label")
    path = item.get("path")
    train_shifts = item.get("train_shifts")
    val_shifts = item.get("val_shifts")
    if not isinstance(period, int) or not isinstance(label, int) or not isinstance(path, str):
        raise ValueError(f"invalid shift-pool shard metadata in {manifest_path}")
    if not isinstance(train_shifts, list) or not all(isinstance(value, int) for value in train_shifts):
        raise ValueError(f"invalid train_shifts in {manifest_path}")
    if not isinstance(val_shifts, list) or not all(isinstance(value, int) for value in val_shifts):
        raise ValueError(f"invalid val_shifts in {manifest_path}")
    return ShiftPoolShardInfo(
        period=period,
        label=label,
        path=path,
        train_shifts=tuple(int(value) for value in train_shifts),
        val_shifts=tuple(int(value) for value in val_shifts),
    )


def load_shift_pool_manifest(path: Path) -> ShiftPoolManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid shift-pool manifest in {path}")

    config = payload.get("config")
    candidate_periods = payload.get("candidate_periods")
    shards_payload = payload.get("shards")
    if not isinstance(config, dict):
        raise ValueError(f"invalid shift-pool config in {path}")
    if not isinstance(candidate_periods, list) or not all(isinstance(value, int) for value in candidate_periods):
        raise ValueError(f"invalid candidate_periods in {path}")
    if not isinstance(shards_payload, list):
        raise ValueError(f"invalid shards payload in {path}")

    shards = tuple(_parse_shift_pool_shard_info(item, path) for item in shards_payload)
    return ShiftPoolManifest(
        manifest_path=path,
        nqubit=int(payload["nqubit"]),
        measurement_count=int(payload["measurement_count"]),
        pool_size=int(payload["pool_size"]),
        pool_multiplier=int(payload["pool_multiplier"]),
        held_out_shifts_per_period=int(payload["held_out_shifts_per_period"]),
        val_draws_per_heldout_shift=int(payload["val_draws_per_heldout_shift"]),
        train_draws_per_epoch=int(payload["train_draws_per_epoch"]),
        seed=int(payload["seed"]),
        exact_support=bool(payload["exact_support"]),
        variant_tag=payload.get("variant_tag"),
        candidate_periods=tuple(int(value) for value in candidate_periods),
        shards=shards,
        config=config,
    )


def _shift_pool_manifest_is_current(path: Path, config: PeriodRecoveryDatasetConfig) -> bool:
    if not path.exists():
        return False
    try:
        manifest = load_shift_pool_manifest(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False

    if manifest.config != serialize_dataset_config(config, split="shift_pool_manifest"):
        return False
    return all((config.cache_root / shard.path).exists() for shard in manifest.shards)


def _load_shift_pool_shard(path: Path) -> tuple[Tensor, int, Tensor]:
    payload = torch.load(path, map_location="cpu")
    bit_pools = payload.get("bit_pools")
    shifts = payload.get("shifts")
    label = payload.get("label")
    if not isinstance(bit_pools, Tensor) or not isinstance(shifts, Tensor) or not isinstance(label, int):
        raise ValueError(f"invalid shift-pool shard at {path}")
    return bit_pools, label, shifts


class PoolBackedPeriodDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        manifest: ShiftPoolManifest,
        *,
        split: str,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.manifest = manifest
        self.split = split
        self.measurement_count = manifest.measurement_count
        self.nqubit = manifest.nqubit
        self.candidate_periods = tuple(manifest.candidate_periods)
        self._shard_cache: OrderedDict[int, tuple[Tensor, int, Tensor]] = OrderedDict()
        self._epoch = 0
        self._max_cached_shards = 2
        self._train_entries: tuple[PoolDrawEntry, ...] = tuple(
            PoolDrawEntry(
                period=shard.period,
                shift=shift,
                label=shard.label,
                draw_index=0,
            )
            for shard in manifest.shards
            for shift in shard.train_shifts
        )
        self.entries: tuple[PoolDrawEntry, ...] = (
            self._train_entries
            if split == "train"
            else tuple(
                PoolDrawEntry(
                    period=shard.period,
                    shift=shift,
                    label=shard.label,
                    draw_index=draw_index,
                )
                for shard in manifest.shards
                for shift in shard.val_shifts
                for draw_index in range(manifest.val_draws_per_heldout_shift)
            )
        )
        self._epoch_entries: tuple[PoolDrawEntry, ...] = self._train_entries
        if split == "train":
            self.set_epoch(0)

    def set_epoch(self, epoch: int, *, batch_size: int | None = None) -> None:
        self._epoch = epoch
        if self.split != "train":
            return
        rng = np.random.default_rng(self.manifest.seed + (epoch + 1) * 1_000_003)
        sampled_entries = [
            self._train_entries[int(rng.integers(0, len(self._train_entries)))]
            for _ in range(self.manifest.train_draws_per_epoch)
        ]
        if batch_size is None or batch_size < 2:
            self._epoch_entries = tuple(sampled_entries)
            return

        entries_by_period: dict[int, list[PoolDrawEntry]] = {}
        for entry in sampled_entries:
            entries_by_period.setdefault(entry.period, []).append(entry)

        chunks: list[list[PoolDrawEntry]] = []
        for entries in entries_by_period.values():
            for start in range(0, len(entries), batch_size):
                chunks.append(entries[start : start + batch_size])
        rng.shuffle(chunks)
        self._epoch_entries = tuple(entry for chunk in chunks for entry in chunk)

    def __len__(self) -> int:
        if self.split == "train":
            return len(self._epoch_entries)
        return len(self.entries)

    def _shard_bit_pool(self, period: int, shift: int) -> tuple[Tensor, int]:
        cached = self._shard_cache.get(period)
        if cached is None:
            shard_path = self.manifest.manifest_path.parent / next(
                shard.path for shard in self.manifest.shards if shard.period == period
            )
            cached = _load_shift_pool_shard(shard_path)
            self._shard_cache[period] = cached
            while len(self._shard_cache) > self._max_cached_shards:
                self._shard_cache.popitem(last=False)
        else:
            self._shard_cache.move_to_end(period)
        bit_pools, label, _ = cached
        return bit_pools[shift], label

    def describe_index(self, index: int) -> PoolDrawEntry:
        if self.split == "train":
            return self._epoch_entries[index]
        return self.entries[index]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        entry = self.describe_index(index)
        bit_pool, label = self._shard_bit_pool(entry.period, entry.shift)
        rng = np.random.default_rng(
            self.manifest.seed
            + entry.period * 100_003
            + entry.shift * 1_009
            + entry.draw_index * 17
            + (self._epoch if self.split == "train" else 0) * 1_000_003
            + index
        )
        row_indices = torch.from_numpy(
            rng.choice(
                self.manifest.pool_size,
                size=self.measurement_count,
                replace=False,
            ).astype(np.int64)
        )
        return (
            bit_pool.index_select(0, row_indices),
            torch.tensor(label, dtype=torch.long),
        )


def _columns_to_bit_matrix(columns: np.ndarray, nqubit: int) -> np.ndarray:
    bit_positions = np.arange(nqubit - 1, -1, -1, dtype=np.int64)
    return ((columns[:, None] >> bit_positions) & 1).astype(np.int8)


def _normalized_distribution(probability: ProbFunc, size: int, shift: int) -> np.ndarray:
    distribution = probability_distribution(probability, size, shift=shift)
    distribution = np.asarray(distribution, dtype=np.float64)
    total = distribution.sum()
    if total <= 0.0:
        raise ValueError("probability distribution must have positive mass")
    normalized = np.asarray(np.clip(distribution / total, 0.0, None), dtype=np.float64)
    normalized /= normalized.sum()
    normalized[-1] = max(0.0, 1.0 - normalized[:-1].sum())
    return np.asarray(normalized / normalized.sum(), dtype=np.float64)


def _sample_bit_matrix_from_distribution(
    distribution: np.ndarray,
    sample_count: int,
    basis_bit_rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = rng.multinomial(sample_count, distribution)
    bit_matrix = np.repeat(
        basis_bit_rows,
        counts.astype(np.int64, copy=False),
        axis=0,
    )
    if bit_matrix.shape[0] != sample_count:
        raise RuntimeError("sampled bit-matrix row count does not match requested sample_count")
    return np.asarray(bit_matrix, dtype=np.int8)


def _torch_distribution_for_period_shift(
    unitary: Tensor,
    period: int,
    shift: int,
    *,
    exact_support: bool,
) -> np.ndarray:
    support_indices = (
        _torch_exact_support_indices(
            int(unitary.shape[1]),
            period,
            shift,
            device=unitary.device,
        )
        if exact_support
        else _torch_surrogate_support_indices(
            int(unitary.shape[1]),
            period,
            shift,
            device=unitary.device,
        )
    )
    probabilities = _torch_probability_vector_from_support(unitary, support_indices)
    normalized = probabilities / probabilities.sum()
    distribution = np.asarray(normalized.detach().cpu().to(torch.float64).numpy(), dtype=np.float64)
    distribution = np.clip(distribution, 0.0, None)
    distribution /= distribution.sum()
    distribution[-1] = max(0.0, 1.0 - distribution[:-1].sum())
    return np.asarray(distribution / distribution.sum(), dtype=np.float64)


def _sample_split_payload(
    sample_count: int,
    config: PeriodRecoveryDatasetConfig,
    probabilities: dict[int, ProbFunc],
    rng: np.random.Generator,
    *,
    split: str,
    period_schedule: Sequence[int] | None = None,
) -> DatasetPayload:
    candidate_periods = config.candidate_periods
    size = 1 << config.nqubit
    distribution_cache: dict[tuple[int, int], np.ndarray] = {}
    basis_bit_rows = _columns_to_bit_matrix(np.arange(size, dtype=np.int64), config.nqubit)

    bit_matrices = np.empty(
        (sample_count, config.measurement_count, config.nqubit),
        dtype=np.int8,
    )
    labels = np.empty(sample_count, dtype=np.int64)
    periods = np.empty(sample_count, dtype=np.int64)
    shifts = np.empty(sample_count, dtype=np.int64)
    resolved_schedule = list(period_schedule) if period_schedule is not None else None
    if resolved_schedule is not None and len(resolved_schedule) != sample_count:
        raise ValueError("period_schedule length must match sample_count")

    for sample_index in range(sample_count):
        if resolved_schedule is None:
            period = int(rng.choice(candidate_periods))
        else:
            period = int(resolved_schedule[sample_index])
        shift = int(rng.integers(0, period))
        cache_key = (period, shift)
        distribution = distribution_cache.get(cache_key)
        if distribution is None:
            distribution = _normalized_distribution(probabilities[period], size, shift)
            distribution_cache[cache_key] = distribution

        bit_matrices[sample_index] = _sample_bit_matrix_from_distribution(
            distribution,
            config.measurement_count,
            basis_bit_rows,
            rng,
        )
        labels[sample_index] = period
        periods[sample_index] = period
        shifts[sample_index] = shift

    return {
        "bit_matrices": torch.from_numpy(bit_matrices),
        "labels": torch.from_numpy(labels),
        "periods": torch.from_numpy(periods),
        "shifts": torch.from_numpy(shifts),
        "candidate_periods": torch.tensor(candidate_periods, dtype=torch.long),
        "config": serialize_dataset_config(config, split=split),
    }


def _sample_period_schedule(
    candidate_periods: Sequence[int],
    sample_count: int,
    rng: np.random.Generator,
) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not candidate_periods:
        raise ValueError("candidate_periods must not be empty")

    period_pool = np.asarray(candidate_periods, dtype=np.int64)
    periods: list[int] = []
    full_cycles, remainder = divmod(sample_count, len(period_pool))
    for _ in range(full_cycles):
        periods.extend(int(period) for period in rng.permutation(period_pool))
    if remainder:
        periods.extend(int(period) for period in rng.choice(period_pool, size=remainder, replace=False))
    return periods


def load_cached_dataset(path: Path) -> CachedPeriodDataset:
    payload = torch.load(path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"invalid cached dataset config in {path}")

    tensors = {}
    for key in ("bit_matrices", "labels", "periods", "shifts", "candidate_periods"):
        value = payload.get(key)
        if not isinstance(value, Tensor):
            raise ValueError(f"invalid cached dataset tensor '{key}' in {path}")
        tensors[key] = value

    return CachedPeriodDataset(
        bit_matrices=tensors["bit_matrices"],
        labels=tensors["labels"],
        periods=tensors["periods"],
        shifts=tensors["shifts"],
        candidate_periods=tensors["candidate_periods"],
        config=config,
    )


def resolve_train_device() -> torch.device:
    requested = os.environ.get("ALTQFT_TRAIN_DEVICE", "auto").strip().lower()
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested for period recovery but CUDA is unavailable")
        return torch.device(requested)
    return torch.device(resolve_compute_device(requested))


def configure_train_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = True


def move_batch_to_device(
    bit_matrices: Tensor,
    labels: Tensor,
    device: torch.device,
) -> TrainBatch:
    non_blocking = device.type == "cuda"
    return (
        bit_matrices.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        labels.to(device=device, dtype=torch.long, non_blocking=non_blocking),
    )


def autocast_context(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _dataset_is_current(
    path: Path,
    config: PeriodRecoveryDatasetConfig,
    *,
    split: str,
) -> bool:
    if not path.exists():
        return False

    try:
        cached = load_cached_dataset(path)
    except (OSError, RuntimeError, ValueError):
        return False

    return cached.config == serialize_dataset_config(config, split=split)


def _generate_shift_pool_dataset(
    config: PeriodRecoveryDatasetConfig,
    optimized_ph1: OptimizedPH1Artifact,
) -> PeriodRecoveryDatasetArtifacts:
    config.cache_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    size = 1 << config.nqubit
    basis_bit_rows = _columns_to_bit_matrix(np.arange(size, dtype=np.int64), config.nqubit)
    cache_device = resolve_compute_device(config.cache_device)
    unitary = _torch_unitary(optimized_ph1.circuit, cache_device)
    shards: list[ShiftPoolShardInfo] = []

    for period in config.candidate_periods:
        period_start = time.perf_counter()
        bit_pools = np.empty((period, config.pool_size, config.nqubit), dtype=np.int8)
        for shift in range(period):
            distribution = _torch_distribution_for_period_shift(
                unitary,
                period,
                shift,
                exact_support=config.exact_support,
            )
            bit_pools[shift] = _sample_bit_matrix_from_distribution(
                distribution,
                config.pool_size,
                basis_bit_rows,
                rng,
            )

        val_shifts = _held_out_shifts_for_period(
            period,
            count=config.held_out_shifts_per_period,
            seed=config.seed,
        )
        val_shift_set = set(val_shifts)
        train_shifts = tuple(shift for shift in range(period) if shift not in val_shift_set)
        shard_path = config.period_shard_path(period)
        torch.save(
            {
                "period": period,
                "label": period,
                "shifts": torch.arange(period, dtype=torch.long),
                "bit_pools": torch.from_numpy(bit_pools),
            },
            shard_path,
        )
        shards.append(
            ShiftPoolShardInfo(
                period=period,
                label=period,
                path=shard_path.name,
                train_shifts=train_shifts,
                val_shifts=val_shifts,
            )
        )
        print(
            "cache_shard "
            f"period={period} "
            f"rows_per_shift={config.pool_size} "
            f"shift_count={period} "
            f"elapsed_seconds={time.perf_counter() - period_start:.2f} "
            f"path={shard_path}",
            flush=True,
        )

    manifest_payload = _serialize_shift_pool_manifest(config, shards)
    config.manifest_path.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    print(
        "cache_manifest "
        f"path={config.manifest_path} "
        f"period_count={len(shards)}",
        flush=True,
    )
    return PeriodRecoveryDatasetArtifacts(
        candidate_periods=config.candidate_periods,
        train_path=config.train_path,
        val_path=config.val_path,
        cache_mode=config.cache_mode,
        manifest_path=config.manifest_path,
    )


def generate_period_recovery_dataset(
    config: PeriodRecoveryDatasetConfig,
    optimized_ph1: OptimizedPH1Artifact,
    *,
    regenerate: bool = False,
) -> PeriodRecoveryDatasetArtifacts:
    if optimized_ph1.nqubit != config.nqubit:
        raise ValueError("optimized PH1 artifact nqubit does not match dataset config")
    if optimized_ph1.period_range != config.candidate_periods:
        raise ValueError("optimized PH1 artifact period_range does not match dataset config")
    if optimized_ph1.exact_support != config.exact_support:
        raise ValueError("optimized PH1 exact_support does not match dataset config")

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    if config.cache_mode == "shift_pool":
        manifest_is_current = _shift_pool_manifest_is_current(config.manifest_path, config)
        if not regenerate and manifest_is_current:
            return PeriodRecoveryDatasetArtifacts(
                candidate_periods=config.candidate_periods,
                train_path=config.train_path,
                val_path=config.val_path,
                cache_mode=config.cache_mode,
                manifest_path=config.manifest_path,
            )
        return _generate_shift_pool_dataset(config, optimized_ph1)

    train_is_current = _dataset_is_current(config.train_path, config, split="train")
    val_is_current = _dataset_is_current(config.val_path, config, split="val")
    if not regenerate and train_is_current and val_is_current:
        return PeriodRecoveryDatasetArtifacts(
            candidate_periods=config.candidate_periods,
            train_path=config.train_path,
            val_path=config.val_path,
            cache_mode=config.cache_mode,
            manifest_path=None,
        )

    rng = np.random.default_rng(config.seed)
    probabilities = {
        period: make_prob(
            optimized_ph1.circuit,
            period,
            exact_support=config.exact_support,
        )
        for period in config.candidate_periods
    }
    train_period_schedule = (
        _sample_period_schedule(config.candidate_periods, config.num_train_samples, rng)
        if config.stratify_periods
        else None
    )
    if config.stratify_periods and train_period_schedule is not None:
        covered_periods = sorted(set(train_period_schedule))
        if len(covered_periods) < len(config.candidate_periods):
            val_period_pool = covered_periods
        else:
            val_period_pool = config.candidate_periods
        val_period_schedule = _sample_period_schedule(val_period_pool, config.num_val_samples, rng)
    else:
        val_period_schedule = None

    train_payload = _sample_split_payload(
        config.num_train_samples,
        config,
        probabilities,
        rng,
        split="train",
        period_schedule=train_period_schedule,
    )
    val_payload = _sample_split_payload(
        config.num_val_samples,
        config,
        probabilities,
        rng,
        split="val",
        period_schedule=val_period_schedule,
    )
    torch.save(train_payload, config.train_path)
    torch.save(val_payload, config.val_path)
    return PeriodRecoveryDatasetArtifacts(
        candidate_periods=config.candidate_periods,
        train_path=config.train_path,
        val_path=config.val_path,
        cache_mode=config.cache_mode,
        manifest_path=None,
    )


def create_dataloader(
    dataset: CachedPeriodDataset | PoolBackedPeriodDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[Tensor, Tensor]]:
    tensor_dataset = (
        cast(
            Dataset[tuple[Tensor, Tensor]],
            TensorDataset(dataset.bit_matrices, dataset.labels),
        )
        if isinstance(dataset, CachedPeriodDataset)
        else dataset
    )
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle if isinstance(dataset, CachedPeriodDataset) else False,
        pin_memory=device.type == "cuda",
    )


def period_prediction_accuracy(
    predicted_periods: Tensor,
    periods: Tensor,
    k: int,
) -> float:
    width = min(k, predicted_periods.shape[1])
    correct = predicted_periods[:, :width].eq(periods.unsqueeze(1)).any(dim=1)
    return float(correct.to(torch.float32).mean().item())


def _evaluate_model(
    model: DeepSetPeriodPredictor,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    top_k: int,
    label_smoothing: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_topk = 0.0
    total_items = 0

    with torch.inference_mode():
        for bit_matrices, periods in dataloader:
            bit_matrices, periods = move_batch_to_device(bit_matrices, periods, device)
            with autocast_context(device):
                pooled = model.pooled_features(bit_matrices)
                token_logits = model.decode_teacher_forced(pooled, periods)
                loss = period_token_loss(
                    token_logits,
                    periods,
                    label_smoothing=label_smoothing,
                )
                predicted_periods, _, _ = model.decode_topk_from_pooled_features(
                    pooled,
                    top_k,
                )
            batch_size = int(periods.shape[0])
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_top1 += (
                period_prediction_accuracy(predicted_periods, periods, 1) * batch_size
            )
            total_topk += (
                period_prediction_accuracy(predicted_periods, periods, top_k)
                * batch_size
            )

    if total_items == 0:
        raise RuntimeError("dataloader is empty")
    return total_loss / total_items, total_top1 / total_items, total_topk / total_items


def save_history(
    config: PeriodRecoveryTrainConfig,
    history: list[PeriodRecoveryEpochResult],
    dataset_artifacts: PeriodRecoveryDatasetArtifacts,
    optimized_ph1: OptimizedPH1Artifact,
) -> None:
    payload = {
        "config": serialize_train_config(config),
        "history": [asdict(item) for item in history],
        "train_dataset_path": str(dataset_artifacts.train_path),
        "val_dataset_path": str(dataset_artifacts.val_path),
        "phase_path": str(optimized_ph1.phase_path),
    }
    config.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_better_result(
    candidate: PeriodRecoveryEpochResult,
    incumbent: PeriodRecoveryEpochResult | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (candidate.val_top1, candidate.val_topk, -candidate.val_loss, -candidate.epoch)
    incumbent_key = (incumbent.val_top1, incumbent.val_topk, -incumbent.val_loss, -incumbent.epoch)
    return candidate_key > incumbent_key


def _load_period_recovery_datasets(
    dataset_artifacts: PeriodRecoveryDatasetArtifacts,
) -> tuple[
    CachedPeriodDataset | PoolBackedPeriodDataset,
    CachedPeriodDataset | PoolBackedPeriodDataset,
    list[int],
]:
    if dataset_artifacts.cache_mode == 'shift_pool':
        manifest_path = dataset_artifacts.manifest_path or dataset_artifacts.train_path
        manifest = load_shift_pool_manifest(manifest_path)
        train_dataset = PoolBackedPeriodDataset(manifest, split='train')
        val_dataset = PoolBackedPeriodDataset(manifest, split='val')
        candidate_periods = list(manifest.candidate_periods)
        if candidate_periods != dataset_artifacts.candidate_periods:
            raise ValueError('shift-pool manifest candidate periods do not match dataset artifacts')
        return train_dataset, val_dataset, candidate_periods

    cached_train = load_cached_dataset(dataset_artifacts.train_path)
    cached_val = load_cached_dataset(dataset_artifacts.val_path)
    candidate_periods = cached_train.candidate_periods.tolist()
    if candidate_periods != dataset_artifacts.candidate_periods:
        raise ValueError('cached train dataset candidate periods do not match dataset artifacts')
    if cached_val.candidate_periods.tolist() != candidate_periods:
        raise ValueError('cached val dataset candidate periods do not match train dataset')
    return cached_train, cached_val, candidate_periods


def _train_period_recovery_epoch(
    model: DeepSetPeriodPredictor,
    train_loader: DataLoader[TrainBatch],
    *,
    train_dataset: CachedPeriodDataset | PoolBackedPeriodDataset,
    optimizer: Adam,
    device: torch.device,
    top_k: int,
    label_smoothing: float,
    epoch: int,
    train_batch_size: int,
) -> tuple[float, float, float]:
    if isinstance(train_dataset, PoolBackedPeriodDataset):
        train_dataset.set_epoch(epoch - 1, batch_size=train_batch_size)

    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_topk = 0.0
    total_items = 0

    for bit_matrices, periods in train_loader:
        bit_matrices, periods = move_batch_to_device(bit_matrices, periods, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            pooled = model.pooled_features(bit_matrices)
            token_logits = model.decode_teacher_forced(pooled, periods)
            loss = period_token_loss(
                token_logits,
                periods,
                label_smoothing=label_smoothing,
            )
            with torch.no_grad():
                predicted_periods, _, _ = model.decode_topk_from_pooled_features(
                    pooled.detach(),
                    top_k,
                )
        loss.backward()
        optimizer.step()

        batch_items = int(periods.shape[0])
        total_items += batch_items
        total_loss += float(loss.item()) * batch_items
        total_top1 += (
            period_prediction_accuracy(predicted_periods, periods, 1) * batch_items
        )
        total_topk += (
            period_prediction_accuracy(predicted_periods, periods, top_k)
            * batch_items
        )

    if total_items == 0:
        raise RuntimeError('train dataloader is empty')

    return (
        total_loss / total_items,
        total_top1 / total_items,
        total_topk / total_items,
    )


def _save_period_recovery_checkpoint(
    config: PeriodRecoveryTrainConfig,
    *,
    best_checkpoint: PeriodRecoveryCheckpoint,
    model: DeepSetPeriodPredictor,
    candidate_periods: list[int],
) -> None:
    torch.save(
        {
            'state_dict': best_checkpoint.state_dict,
            'candidate_periods': candidate_periods,
            'period_min': model.period_min,
            'period_max': model.period_max,
            'bit_width': model.bit_width,
            'token_bits': TOKEN_BITS,
            'token_count': model.token_count,
            'beam_width': model.beam_width,
            'decoder_type': DECODER_TYPE,
            'model_architecture': model.architecture,
            'num_periods': model.num_periods,
            'selected_epoch': best_checkpoint.result.epoch,
            'config': serialize_train_config(config),
        },
        config.model_path,
    )


def train_period_recovery(
    config: PeriodRecoveryTrainConfig,
    dataset_artifacts: PeriodRecoveryDatasetArtifacts,
    optimized_ph1: OptimizedPH1Artifact,
) -> PeriodRecoveryTrainArtifacts:
    if optimized_ph1.nqubit != config.nqubit:
        raise ValueError('optimized PH1 artifact nqubit does not match train config')
    if dataset_artifacts.cache_mode != config.dataset_mode:
        raise ValueError('dataset artifacts cache mode does not match train config')

    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(LOGGER_NAME, config.log_path)
    set_random_seed(config.seed)
    device = resolve_train_device()
    configure_train_backend(device)
    logger.info('start period recovery training with config=%s', json.dumps(serialize_train_config(config)))
    logger.info(
        'period recovery runtime device=%s amp=%s pin_memory=%s requested_device=%s',
        device,
        device.type == 'cuda',
        device.type == 'cuda',
        os.environ.get('ALTQFT_TRAIN_DEVICE', 'auto'),
    )
    if device.type == 'cpu' and torch.version.cuda is None:
        logger.info('torch build has no CUDA runtime; period recovery will run on CPU')

    train_dataset, val_dataset, candidate_periods = _load_period_recovery_datasets(dataset_artifacts)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        device=device,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        device=device,
    )
    _log_dataset_diagnostics(logger, 'train', summarize_bitmatrix_dataset(train_dataset))
    _log_dataset_diagnostics(logger, 'val', summarize_bitmatrix_dataset(val_dataset))

    model = DeepSetPeriodPredictor(
        config.nqubit,
        len(candidate_periods),
        period_min=candidate_periods[0],
        period_max=candidate_periods[-1],
        dropout=config.dropout,
        beam_width=DEFAULT_BEAM_WIDTH,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[PeriodRecoveryEpochResult] = []
    best_checkpoint: PeriodRecoveryCheckpoint | None = None
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_top1, train_topk = _train_period_recovery_epoch(
            model,
            train_loader,
            train_dataset=train_dataset,
            optimizer=optimizer,
            device=device,
            top_k=config.top_k,
            label_smoothing=config.label_smoothing,
            epoch=epoch,
            train_batch_size=config.batch_size,
        )
        val_loss, val_top1, val_topk = _evaluate_model(
            model,
            val_loader,
            device=device,
            top_k=config.top_k,
            label_smoothing=config.label_smoothing,
        )
        result = PeriodRecoveryEpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_top1=train_top1,
            train_topk=train_topk,
            val_loss=val_loss,
            val_top1=val_top1,
            val_topk=val_topk,
        )
        history.append(result)
        incumbent = best_checkpoint.result if best_checkpoint is not None else None
        if _is_better_result(result, incumbent):
            best_checkpoint = PeriodRecoveryCheckpoint(
                result=result,
                state_dict=snapshot_model_state(model),
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
            logger.info(
                'epoch=%s/%s train_loss=%.8f train_top1=%.4f train_top%d=%.4f '
                'val_loss=%.8f val_top1=%.4f val_top%d=%.4f',
                epoch,
                config.epochs,
                result.train_loss,
                result.train_top1,
                config.top_k,
                result.train_topk,
                result.val_loss,
                result.val_top1,
                config.top_k,
                result.val_topk,
            )
        if epoch >= config.min_epochs and stale_epochs >= config.early_stopping_patience:
            logger.info(
                'early stopping at epoch=%s after %s stale epoch(s); best_epoch=%s best_val_top1=%.4f best_val_top%d=%.4f',
                epoch,
                stale_epochs,
                best_checkpoint.result.epoch if best_checkpoint is not None else epoch,
                best_checkpoint.result.val_top1 if best_checkpoint is not None else result.val_top1,
                config.top_k,
                best_checkpoint.result.val_topk if best_checkpoint is not None else result.val_topk,
            )
            break

    if best_checkpoint is None:
        raise RuntimeError('training did not produce any checkpoint')

    _save_period_recovery_checkpoint(
        config,
        best_checkpoint=best_checkpoint,
        model=model,
        candidate_periods=candidate_periods,
    )
    save_history(config, history, dataset_artifacts, optimized_ph1)
    last_epoch = history[-1]
    selected_epoch = best_checkpoint.result
    stopped_early = len(history) < config.epochs
    logger.info(
        (
            'training finished selected_epoch=%s selected_val_top1=%.4f selected_val_top%d=%.4f '
            'last_epoch=%s last_val_top1=%.4f last_val_top%d=%.4f stopped_early=%s'
        ),
        selected_epoch.epoch,
        selected_epoch.val_top1,
        config.top_k,
        selected_epoch.val_topk,
        last_epoch.epoch,
        last_epoch.val_top1,
        config.top_k,
        last_epoch.val_topk,
        stopped_early,
    )
    return PeriodRecoveryTrainArtifacts(
        history=history,
        top_k=config.top_k,
        selected_epoch=selected_epoch.epoch,
        selected_train_top1=selected_epoch.train_top1,
        selected_train_topk=selected_epoch.train_topk,
        selected_val_top1=selected_epoch.val_top1,
        selected_val_topk=selected_epoch.val_topk,
        last_epoch=last_epoch.epoch,
        last_train_top1=last_epoch.train_top1,
        last_train_topk=last_epoch.train_topk,
        last_val_top1=last_epoch.val_top1,
        last_val_topk=last_epoch.val_topk,
        stopped_early=stopped_early,
        model_path=config.model_path,
        history_path=config.history_path,
        log_path=config.log_path,
        train_dataset_path=dataset_artifacts.train_path,
        val_dataset_path=dataset_artifacts.val_path,
        phase_path=optimized_ph1.phase_path,
    )
