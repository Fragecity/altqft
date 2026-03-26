from __future__ import annotations

import json
import logging
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
from altqft.nn.process_qc import ProbFunc, make_prob, probability_distribution

LOGGER_NAME = "altqft.nn.period_recovery"
DatasetPayload = dict[str, Tensor | dict[str, Any]]


@dataclass(slots=True)
class PeriodRecoveryDatasetConfig:
    nqubit: int
    measurement_count: int
    num_train_samples: int
    num_val_samples: int
    seed: int = 7
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
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
        if self.fi_epochs < 1:
            raise ValueError("fi_epochs must be positive")
        if self.fi_log_interval < 1:
            raise ValueError("fi_log_interval must be positive")

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
class PeriodRecoveryTrainArtifacts:
    history: list[PeriodRecoveryEpochResult]
    top_k: int
    final_train_top1: float
    final_train_topk: float
    final_val_top1: float
    final_val_topk: float
    model_path: Path
    history_path: Path
    log_path: Path
    train_dataset_path: Path
    val_dataset_path: Path
    phase_path: Path


class DeepSetPeriodPredictor(nn.Module):
    def __init__(self, nqubit: int, num_periods: int) -> None:
        super().__init__()
        feature_dim = 16 * nqubit
        self.phi = nn.Sequential(
            nn.Linear(nqubit, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_periods),
        )

    def forward(self, bit_matrices: Tensor) -> Tensor:
        features = self.phi(bit_matrices.to(torch.float32))
        pooled = features.sum(dim=1)
        return cast(Tensor, self.head(pooled))


def serialize_dataset_config(
    config: PeriodRecoveryDatasetConfig,
    *,
    split: str,
) -> dict[str, int | str | list[int]]:
    return {
        "nqubit": config.nqubit,
        "measurement_count": config.measurement_count,
        "num_train_samples": config.num_train_samples,
        "num_val_samples": config.num_val_samples,
        "seed": config.seed,
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


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    for sample_index in range(sample_count):
        period = int(rng.choice(candidate_periods))
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
    if not regenerate and _dataset_is_current(config.train_path, config, split="train") and _dataset_is_current(
        config.val_path,
        config,
        split="val",
    ):
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

    train_payload = _sample_split_payload(
        config.num_train_samples,
        config,
        probabilities,
        rng,
        split="train",
    )
    val_payload = _sample_split_payload(
        config.num_val_samples,
        config,
        probabilities,
        rng,
        split="val",
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
) -> DataLoader[tuple[Tensor, Tensor]]:
    tensor_dataset = cast(
        Dataset[tuple[Tensor, Tensor]],
        TensorDataset(dataset.bit_matrices, dataset.labels),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def topk_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    effective_k = min(k, logits.shape[1])
    topk_indices = logits.topk(effective_k, dim=1).indices
    correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.to(torch.float32).mean().item())


def _evaluate_model(
    model: DeepSetPeriodPredictor,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    *,
    top_k: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_topk = 0.0
    total_items = 0

    with torch.no_grad():
        for bit_matrices, labels in dataloader:
            logits = model(bit_matrices)
            loss = loss_fn(logits, labels)
            batch_size = int(labels.shape[0])
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_top1 += topk_accuracy(logits, labels, 1) * batch_size
            total_topk += topk_accuracy(logits, labels, top_k) * batch_size

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


def train_period_recovery(
    config: PeriodRecoveryTrainConfig,
    dataset_artifacts: PeriodRecoveryDatasetArtifacts,
    optimized_ph1: OptimizedPH1Artifact,
) -> PeriodRecoveryTrainArtifacts:
    if optimized_ph1.nqubit != config.nqubit:
        raise ValueError("optimized PH1 artifact nqubit does not match train config")

    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(config.log_path)
    set_random_seed(config.seed)
    logger.info("start period recovery training with config=%s", json.dumps(serialize_train_config(config)))

    train_dataset = load_cached_dataset(dataset_artifacts.train_path)
    val_dataset = load_cached_dataset(dataset_artifacts.val_path)
    candidate_periods = train_dataset.candidate_periods.tolist()
    if candidate_periods != dataset_artifacts.candidate_periods:
        raise ValueError("cached train dataset candidate periods do not match dataset artifacts")
    if val_dataset.candidate_periods.tolist() != candidate_periods:
        raise ValueError("cached val dataset candidate periods do not match train dataset")

    train_loader = create_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = DeepSetPeriodPredictor(config.nqubit, len(candidate_periods))
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[PeriodRecoveryEpochResult] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_top1 = 0.0
        total_topk = 0.0
        total_items = 0

        for bit_matrices, labels in train_loader:
            optimizer.zero_grad()
            logits = model(bit_matrices)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            total_items += batch_size
            total_loss += float(loss.item()) * batch_size
            total_top1 += topk_accuracy(logits.detach(), labels, 1) * batch_size
            total_topk += topk_accuracy(logits.detach(), labels, config.top_k) * batch_size

        if total_items == 0:
            raise RuntimeError("train dataloader is empty")

        val_loss, val_top1, val_topk = _evaluate_model(
            model,
            val_loader,
            loss_fn,
            top_k=config.top_k,
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

    torch.save(
        {
            "state_dict": model.state_dict(),
            "candidate_periods": candidate_periods,
            "config": serialize_train_config(config),
        },
        config.model_path,
    )
    save_history(config, history, dataset_artifacts, optimized_ph1)
    final_epoch = history[-1]
    logger.info(
        "training finished final_val_top1=%.4f final_val_top%d=%.4f",
        final_epoch.val_top1,
        config.top_k,
        final_epoch.val_topk,
    )
    return PeriodRecoveryTrainArtifacts(
        history=history,
        top_k=config.top_k,
        final_train_top1=final_epoch.train_top1,
        final_train_topk=final_epoch.train_topk,
        final_val_top1=final_epoch.val_top1,
        final_val_topk=final_epoch.val_topk,
        model_path=config.model_path,
        history_path=config.history_path,
        log_path=config.log_path,
        train_dataset_path=dataset_artifacts.train_path,
        val_dataset_path=dataset_artifacts.val_path,
        phase_path=optimized_ph1.phase_path,
    )
