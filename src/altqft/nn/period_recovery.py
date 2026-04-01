from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from altqft.nn.optimized_ph1 import OptimizedPH1Artifact
from altqft.nn.periods import build_default_period_range
from altqft.nn.process_qc import (
    ProbFunc,
    make_prob,
    probability_distribution,
    resolve_compute_device,
)
from altqft.nn.runtime import configure_logger, set_random_seed, snapshot_model_state

LOGGER_NAME = "altqft.nn.period_recovery"
DatasetPayload = dict[str, Tensor | dict[str, Any]]
TrainBatch = tuple[Tensor, Tensor]


@dataclass(slots=True)
class PeriodRecoveryDatasetConfig:
    nqubit: int
    measurement_count: int
    num_train_samples: int
    num_val_samples: int
    seed: int = 7
    stratify_periods: bool = True
    dataset_dir: Path = Path("data/period_recovery")

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if self.measurement_count < 1:
            raise ValueError("measurement_count must be positive")
        if self.num_train_samples < 1:
            raise ValueError("num_train_samples must be positive")
        if self.num_val_samples < 1:
            raise ValueError("num_val_samples must be positive")

    @property
    def candidate_periods(self) -> list[int]:
        return build_default_period_range(self.nqubit)

    @property
    def dataset_stem(self) -> str:
        return (
            f"period_recovery_{self.nqubit}q_"
            f"m{self.measurement_count}_"
            f"train{self.num_train_samples}_"
            f"val{self.num_val_samples}_"
            f"seed{self.seed}"
        )

    @property
    def train_path(self) -> Path:
        return self.dataset_dir / f"{self.dataset_stem}_train.pt"

    @property
    def val_path(self) -> Path:
        return self.dataset_dir / f"{self.dataset_stem}_val.pt"


@dataclass(slots=True)
class PeriodRecoveryTrainConfig:
    nqubit: int
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

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
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

    @property
    def run_name(self) -> str:
        return f"period_recovery_{self.nqubit}q"

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


class DeepSetPeriodPredictor(nn.Module):
    def __init__(self, nqubit: int, num_periods: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if num_periods < 1:
            raise ValueError("num_periods must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_periods = num_periods
        self.bit_width = compact_label_bit_width(num_periods)
        feature_dim = 16 * nqubit
        self.phi = nn.Sequential(
            nn.Linear(nqubit, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_periods),
        )

    def forward(self, bit_matrices: Tensor) -> Tensor:
        features_input = (
            bit_matrices
            if bit_matrices.dtype == torch.float32
            else bit_matrices.to(dtype=torch.float32)
        )
        features = self.phi(features_input)
        pooled = features.mean(dim=1)
        return cast(Tensor, self.head(pooled))

    def predict_topk_periods(
        self,
        bit_matrices: Tensor,
        candidate_periods: Sequence[int],
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(candidate_periods) != self.num_periods:
            raise ValueError("candidate_periods length does not match model output space")
        logits = self(bit_matrices)
        return decode_topk_periods_from_class_logits(logits, candidate_periods, k)


def compact_label_bit_width(num_periods: int) -> int:
    if num_periods < 1:
        raise ValueError("num_periods must be positive")
    return max(1, (num_periods - 1).bit_length())


def class_indices_to_bits(indices: Tensor, bit_width: int) -> Tensor:
    if bit_width < 1:
        raise ValueError("bit_width must be positive")
    shifts = torch.arange(bit_width - 1, -1, -1, device=indices.device, dtype=torch.long)
    return ((indices.to(torch.long).unsqueeze(-1) >> shifts) & 1).to(torch.long)


def period_bit_loss(bit_logits: Tensor, labels: Tensor) -> Tensor:
    target_bits = class_indices_to_bits(labels, bit_logits.shape[1]).reshape(-1)
    flattened_logits = bit_logits.reshape(-1, 2)
    return cast(Tensor, nn.functional.cross_entropy(flattened_logits, target_bits))


def period_class_loss(logits: Tensor, labels: Tensor, *, label_smoothing: float = 0.0) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("class logits must have shape (batch, num_classes)")
    return cast(
        Tensor,
        nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing),
    )


def decode_topk_class_indices(
    bit_logits: Tensor,
    k: int,
    *,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    if k < 1:
        raise ValueError("k must be positive")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if bit_logits.ndim != 3 or bit_logits.shape[2] != 2:
        raise ValueError("bit_logits must have shape (batch, bit_width, 2)")

    batch_size, bit_width, _ = bit_logits.shape
    log_probs = bit_logits.log_softmax(dim=-1)
    beam_width = min(k, num_classes)
    bit_values = torch.arange(2, device=bit_logits.device, dtype=torch.long).view(1, 1, 2)
    beam_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=bit_logits.device)
    beam_scores = torch.zeros((batch_size, 1), dtype=bit_logits.dtype, device=bit_logits.device)

    for bit_index in range(bit_width):
        expanded_indices = (beam_indices.unsqueeze(-1) << 1) | bit_values
        expanded_scores = beam_scores.unsqueeze(-1) + log_probs[:, bit_index, :].unsqueeze(1)
        remaining_bits = bit_width - bit_index - 1
        valid_prefixes = (expanded_indices << remaining_bits) < num_classes
        expanded_scores = expanded_scores.masked_fill(~valid_prefixes, float("-inf"))

        flat_indices = expanded_indices.reshape(batch_size, -1)
        flat_scores = expanded_scores.reshape(batch_size, -1)
        current_width = min(beam_width, flat_scores.shape[1])
        top_scores, top_positions = flat_scores.topk(current_width, dim=1)
        beam_indices = flat_indices.gather(1, top_positions)
        beam_scores = top_scores

    return beam_indices, beam_scores


def decode_topk_periods(
    bit_logits: Tensor,
    candidate_periods: Sequence[int],
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not candidate_periods:
        raise ValueError("candidate_periods must not be empty")

    top_indices, top_scores = decode_topk_class_indices(
        bit_logits,
        k,
        num_classes=len(candidate_periods),
    )
    candidate_tensor = torch.tensor(candidate_periods, dtype=torch.long, device=bit_logits.device)
    top_periods = candidate_tensor[top_indices]
    top_bits = class_indices_to_bits(top_indices, bit_logits.shape[1])
    return top_periods, top_bits, top_scores


def decode_topk_periods_from_class_logits(
    logits: Tensor,
    candidate_periods: Sequence[int],
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError("class logits must have shape (batch, num_classes)")
    if not candidate_periods:
        raise ValueError("candidate_periods must not be empty")
    if logits.shape[1] != len(candidate_periods):
        raise ValueError("candidate_periods length does not match class logits width")

    beam_width = min(k, len(candidate_periods))
    top_scores, top_indices = logits.log_softmax(dim=1).topk(beam_width, dim=1)
    candidate_tensor = torch.tensor(candidate_periods, dtype=torch.long, device=logits.device)
    top_periods = candidate_tensor[top_indices]
    top_bits = class_indices_to_bits(top_indices, compact_label_bit_width(len(candidate_periods)))
    return top_periods, top_bits, top_scores


def serialize_dataset_config(
    config: PeriodRecoveryDatasetConfig,
    *,
    split: str,
) -> dict[str, int | str | bool | list[int]]:
    return {
        "nqubit": config.nqubit,
        "measurement_count": config.measurement_count,
        "num_train_samples": config.num_train_samples,
        "num_val_samples": config.num_val_samples,
        "seed": config.seed,
        "stratify_periods": config.stratify_periods,
        "dataset_dir": str(config.dataset_dir),
        "candidate_periods": config.candidate_periods,
        "split": split,
    }


def serialize_train_config(config: PeriodRecoveryTrainConfig) -> dict[str, int | float | str | bool]:
    return {
        "nqubit": config.nqubit,
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
    }


def summarize_bitmatrix_dataset(
    dataset: CachedPeriodDataset,
    *,
    measurement_count: int | None = None,
) -> PeriodRecoveryDataDiagnostics:
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


def _columns_to_bit_matrix(columns: np.ndarray, nqubit: int) -> np.ndarray:
    bit_positions = np.arange(nqubit - 1, -1, -1, dtype=np.int64)
    return ((columns[:, None] >> bit_positions) & 1).astype(np.int8)


def _normalized_distribution(probability: ProbFunc, size: int, shift: int) -> np.ndarray:
    distribution = probability_distribution(probability, size, shift=shift)
    total = distribution.sum()
    if total <= 0.0:
        raise ValueError("probability distribution must have positive mass")
    return np.asarray(distribution / total, dtype=np.float64)


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
    class_lookup = {period: index for index, period in enumerate(candidate_periods)}
    size = 1 << config.nqubit
    distribution_cache: dict[tuple[int, int], np.ndarray] = {}

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

        columns = rng.choice(size, size=config.measurement_count, p=distribution)
        bit_matrices[sample_index] = _columns_to_bit_matrix(columns, config.nqubit)
        labels[sample_index] = class_lookup[period]
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


def autocast_context(device: torch.device):
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

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    train_is_current = _dataset_is_current(config.train_path, config, split="train")
    val_is_current = _dataset_is_current(config.val_path, config, split="val")
    if not regenerate and train_is_current and val_is_current:
        return PeriodRecoveryDatasetArtifacts(
            candidate_periods=config.candidate_periods,
            train_path=config.train_path,
            val_path=config.val_path,
        )

    rng = np.random.default_rng(config.seed)
    probabilities = {
        period: make_prob(optimized_ph1.circuit, period)
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
    )


def create_dataloader(
    dataset: CachedPeriodDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[Tensor, Tensor]]:
    tensor_dataset = cast(
        Dataset[tuple[Tensor, Tensor]],
        TensorDataset(dataset.bit_matrices, dataset.labels),
    )
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def topk_accuracy(logits: Tensor, labels: Tensor, k: int, *, num_classes: int) -> float:
    if logits.ndim == 2:
        topk_indices = logits.topk(min(k, num_classes), dim=1).indices
    else:
        topk_indices, _ = decode_topk_class_indices(logits, k, num_classes=num_classes)
    correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.to(torch.float32).mean().item())


def _evaluate_model(
    model: DeepSetPeriodPredictor,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    num_classes: int,
    top_k: int,
    label_smoothing: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_topk = 0.0
    total_items = 0

    with torch.inference_mode():
        for bit_matrices, labels in dataloader:
            bit_matrices, labels = move_batch_to_device(bit_matrices, labels, device)
            with autocast_context(device):
                logits = model(bit_matrices)
                loss = period_class_loss(
                    logits,
                    labels,
                    label_smoothing=label_smoothing,
                )
            batch_size = int(labels.shape[0])
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_top1 += topk_accuracy(logits, labels, 1, num_classes=num_classes) * batch_size
            total_topk += topk_accuracy(logits, labels, top_k, num_classes=num_classes) * batch_size

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


def train_period_recovery(
    config: PeriodRecoveryTrainConfig,
    dataset_artifacts: PeriodRecoveryDatasetArtifacts,
    optimized_ph1: OptimizedPH1Artifact,
) -> PeriodRecoveryTrainArtifacts:
    if optimized_ph1.nqubit != config.nqubit:
        raise ValueError("optimized PH1 artifact nqubit does not match train config")

    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(LOGGER_NAME, config.log_path)
    set_random_seed(config.seed)
    device = resolve_train_device()
    configure_train_backend(device)
    logger.info("start period recovery training with config=%s", json.dumps(serialize_train_config(config)))
    logger.info(
        "period recovery runtime device=%s amp=%s pin_memory=%s requested_device=%s",
        device,
        device.type == "cuda",
        device.type == "cuda",
        os.environ.get("ALTQFT_TRAIN_DEVICE", "auto"),
    )
    if device.type == "cpu" and torch.version.cuda is None:
        logger.info("torch build has no CUDA runtime; period recovery will run on CPU")

    train_dataset = load_cached_dataset(dataset_artifacts.train_path)
    val_dataset = load_cached_dataset(dataset_artifacts.val_path)
    candidate_periods = train_dataset.candidate_periods.tolist()
    if candidate_periods != dataset_artifacts.candidate_periods:
        raise ValueError("cached train dataset candidate periods do not match dataset artifacts")
    if val_dataset.candidate_periods.tolist() != candidate_periods:
        raise ValueError("cached val dataset candidate periods do not match train dataset")

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
    _log_dataset_diagnostics(logger, "train", summarize_bitmatrix_dataset(train_dataset))
    _log_dataset_diagnostics(logger, "val", summarize_bitmatrix_dataset(val_dataset))

    model = DeepSetPeriodPredictor(
        config.nqubit,
        len(candidate_periods),
        dropout=config.dropout,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    num_classes = len(candidate_periods)

    history: list[PeriodRecoveryEpochResult] = []
    best_checkpoint: PeriodRecoveryCheckpoint | None = None
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_top1 = 0.0
        total_topk = 0.0
        total_items = 0

        for bit_matrices, labels in train_loader:
            bit_matrices, labels = move_batch_to_device(bit_matrices, labels, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(bit_matrices)
                loss = period_class_loss(
                    logits,
                    labels,
                    label_smoothing=config.label_smoothing,
                )
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_top1 += topk_accuracy(logits.detach(), labels, 1, num_classes=num_classes) * batch_size
            total_topk += topk_accuracy(logits.detach(), labels, config.top_k, num_classes=num_classes) * batch_size

        if total_items == 0:
            raise RuntimeError("train dataloader is empty")

        val_loss, val_top1, val_topk = _evaluate_model(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            top_k=config.top_k,
            label_smoothing=config.label_smoothing,
        )
        result = PeriodRecoveryEpochResult(
            epoch=epoch,
            train_loss=total_loss / total_items,
            train_top1=total_top1 / total_items,
            train_topk=total_topk / total_items,
            val_loss=val_loss,
            val_top1=val_top1,
            val_topk=val_topk,
        )
        history.append(result)
        if _is_better_result(result, best_checkpoint.result if best_checkpoint is not None else None):
            best_checkpoint = PeriodRecoveryCheckpoint(
                result=result,
                state_dict=snapshot_model_state(model),
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
            logger.info(
                "epoch=%s/%s train_loss=%.8f train_top1=%.4f train_top%d=%.4f "
                "val_loss=%.8f val_top1=%.4f val_top%d=%.4f",
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
                "early stopping at epoch=%s after %s stale epoch(s); best_epoch=%s best_val_top1=%.4f best_val_top%d=%.4f",
                epoch,
                stale_epochs,
                best_checkpoint.result.epoch if best_checkpoint is not None else epoch,
                best_checkpoint.result.val_top1 if best_checkpoint is not None else result.val_top1,
                config.top_k,
                best_checkpoint.result.val_topk if best_checkpoint is not None else result.val_topk,
            )
            break

    if best_checkpoint is None:
        raise RuntimeError("training did not produce any checkpoint")

    torch.save(
        {
            "state_dict": best_checkpoint.state_dict,
            "candidate_periods": candidate_periods,
            "bit_width": model.bit_width,
            "num_periods": model.num_periods,
            "selected_epoch": best_checkpoint.result.epoch,
            "config": serialize_train_config(config),
        },
        config.model_path,
    )
    save_history(config, history, dataset_artifacts, optimized_ph1)
    last_epoch = history[-1]
    selected_epoch = best_checkpoint.result
    stopped_early = len(history) < config.epochs
    logger.info(
        (
            "training finished selected_epoch=%s selected_val_top1=%.4f selected_val_top%d=%.4f "
            "last_epoch=%s last_val_top1=%.4f last_val_top%d=%.4f stopped_early=%s"
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
