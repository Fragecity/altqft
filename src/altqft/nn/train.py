from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

from altqft.nn.model import PH1MinFIModel
from altqft.nn.periods import period_range_artifact_suffix
from altqft.nn.runtime import configure_logger, set_random_seed, snapshot_model_state

LOGGER_NAME = "altqft.nn.train"
SerializedConfig = dict[str, int | float | str | list[int]]


@dataclass(slots=True)
class TrainConfig:
    nqubit: int
    period_range: list[int]
    epochs: int = 100
    learning_rate: float = 0.05
    monte_carlo_samples: int = 0
    seed: int = 7
    log_interval: int = 10
    model_dir: Path = Path("model")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model_stem: str = "ph1_min_fi"

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if not self.period_range:
            raise ValueError("period_range must not be empty")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.monte_carlo_samples < 0:
            raise ValueError("monte_carlo_samples must be non-negative")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")

    @property
    def run_name(self) -> str:
        suffix = period_range_artifact_suffix(self.nqubit, self.period_range)
        return f"{self.model_stem}_{self.nqubit}q{suffix}"

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.run_name}.pt"

    @property
    def phase_path(self) -> Path:
        return self.model_dir / f"{self.run_name}_phases.json"

    @property
    def history_path(self) -> Path:
        return self.output_dir / f"{self.run_name}_history.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.log"


@dataclass(slots=True)
class EpochResult:
    epoch: int
    loss: float
    min_fi: float


@dataclass(slots=True)
class TrainArtifacts:
    history: list[EpochResult]
    final_min_fi: float
    best_epoch: int
    best_min_fi: float
    model_path: Path
    phase_path: Path
    history_path: Path
    log_path: Path


@dataclass(slots=True)
class ModelCheckpoint:
    epoch: int
    min_fi: float
    state_dict: dict[str, Tensor]
    phases: list[float]


def serialize_config(config: TrainConfig) -> SerializedConfig:
    return {
        "nqubit": config.nqubit,
        "period_range": config.period_range,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "monte_carlo_samples": config.monte_carlo_samples,
        "seed": config.seed,
        "log_interval": config.log_interval,
        "model_dir": str(config.model_dir),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "model_stem": config.model_stem,
    }


def prepare_output_dirs(config: TrainConfig) -> None:
    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)


def sample_phase_tensor(phase_count: int) -> torch.Tensor:
    return 2 * torch.pi * torch.rand(phase_count, dtype=torch.float32)


def create_model(
    config: TrainConfig,
    init_phases: Sequence[float] | None = None,
) -> PH1MinFIModel:
    return PH1MinFIModel(nqubit=config.nqubit, init_phases=init_phases)


def select_monte_carlo_init_phases(config: TrainConfig) -> tuple[list[float], float]:
    if config.monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive for Monte Carlo init")

    candidate_model = create_model(config)
    best_phases: list[float] | None = None
    best_min_fi = float("-inf")

    with torch.no_grad():
        for _ in range(config.monte_carlo_samples):
            sampled_phases = sample_phase_tensor(candidate_model.phase_count)
            candidate_model.phases.copy_(sampled_phases)
            min_fi_value = float(
                candidate_model(config.period_range).detach().cpu().item()
            )

            if best_phases is None or min_fi_value > best_min_fi:
                best_phases = sampled_phases.detach().cpu().tolist()
                best_min_fi = min_fi_value

    if best_phases is None:
        raise RuntimeError("monte carlo initialization did not produce any candidate")

    return best_phases, best_min_fi


def initialize_model(config: TrainConfig, logger: logging.Logger) -> PH1MinFIModel:
    if config.monte_carlo_samples < 1:
        return create_model(config)

    logger.info(
        "running monte carlo initialization samples=%s",
        config.monte_carlo_samples,
    )
    init_phases, init_min_fi = select_monte_carlo_init_phases(config)
    logger.info(
        "selected monte carlo initialization min_fi=%.8f",
        init_min_fi,
    )
    return create_model(config, init_phases=init_phases)


def create_optimizer(model: PH1MinFIModel, config: TrainConfig) -> Optimizer:
    return Adam(model.parameters(), lr=config.learning_rate)


def checkpoint_model(model: PH1MinFIModel, epoch: int, min_fi: float) -> ModelCheckpoint:
    return ModelCheckpoint(
        epoch=epoch,
        min_fi=min_fi,
        state_dict=snapshot_model_state(model),
        phases=model.export_phases(),
    )


def train_step(
    model: PH1MinFIModel,
    optimizer: Optimizer,
    period_range: list[int],
) -> tuple[float, float]:
    optimizer.zero_grad()
    min_fi_value = model(period_range)
    loss = -min_fi_value
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        updated_min_fi = float(model(period_range).detach().cpu().item())

    return -updated_min_fi, updated_min_fi


def log_epoch(logger: logging.Logger, result: EpochResult, total_epochs: int) -> None:
    logger.info(
        "epoch=%s/%s loss=%.8f min_fi=%.8f",
        result.epoch,
        total_epochs,
        result.loss,
        result.min_fi,
    )


def run_training(
    model: PH1MinFIModel,
    optimizer: Optimizer,
    config: TrainConfig,
    logger: logging.Logger,
) -> tuple[list[EpochResult], ModelCheckpoint]:
    history: list[EpochResult] = []
    best_checkpoint: ModelCheckpoint | None = None

    for epoch in range(1, config.epochs + 1):
        loss_value, min_fi_value = train_step(model, optimizer, config.period_range)
        result = EpochResult(epoch=epoch, loss=loss_value, min_fi=min_fi_value)
        history.append(result)
        if best_checkpoint is None or result.min_fi > best_checkpoint.min_fi:
            best_checkpoint = checkpoint_model(model, epoch, result.min_fi)

        if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
            log_epoch(logger, result, config.epochs)

    if best_checkpoint is None:
        raise RuntimeError("training did not produce any checkpoint")

    return history, best_checkpoint


def save_history(config: TrainConfig, history: list[EpochResult]) -> None:
    payload = {
        "config": serialize_config(config),
        "history": [asdict(item) for item in history],
    }
    config.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_artifacts(config: TrainConfig, checkpoint: ModelCheckpoint) -> None:
    torch.save(checkpoint.state_dict, config.model_path)
    payload = {
        "nqubit": config.nqubit,
        "period_range": config.period_range,
        "phases": checkpoint.phases,
    }
    config.phase_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_training(
    config: TrainConfig,
    history: list[EpochResult],
    best_checkpoint: ModelCheckpoint,
    logger: logging.Logger,
) -> TrainArtifacts:
    if not history:
        raise RuntimeError("training history is empty")

    final_min_fi = history[-1].min_fi
    logger.info(
        "training finished final_min_fi=%.8f best_min_fi=%.8f best_epoch=%s",
        final_min_fi,
        best_checkpoint.min_fi,
        best_checkpoint.epoch,
    )
    return TrainArtifacts(
        history=history,
        final_min_fi=final_min_fi,
        best_epoch=best_checkpoint.epoch,
        best_min_fi=best_checkpoint.min_fi,
        model_path=config.model_path,
        phase_path=config.phase_path,
        history_path=config.history_path,
        log_path=config.log_path,
    )


def train_model(config: TrainConfig) -> TrainArtifacts:
    prepare_output_dirs(config)
    logger = configure_logger(LOGGER_NAME, config.log_path)
    set_random_seed(config.seed)
    logger.info("start training with config=%s", json.dumps(serialize_config(config)))

    model = initialize_model(config, logger)
    optimizer = create_optimizer(model, config)
    history, best_checkpoint = run_training(model, optimizer, config, logger)
    save_model_artifacts(config, best_checkpoint)
    save_history(config, history)
    return summarize_training(config, history, best_checkpoint, logger)
