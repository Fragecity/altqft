from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

from altqft.nn.model import OBJECTIVES, PH1MinFIModel
from altqft.nn.periods import period_range_artifact_suffix
from altqft.nn.process_qc import resolve_compute_device
from altqft.nn.runtime import configure_logger, set_random_seed, snapshot_model_state

LOGGER_NAME = "altqft.nn.train"
SerializedConfig = dict[str, int | float | str | bool | list[int] | None]


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
    objective: str = "min_fi"
    exact_support: bool = False
    variant_tag: str | None = None
    shift_ce_eps: float = 1e-12
    train_device: str = "auto"

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
        if self.objective not in OBJECTIVES:
            supported = ", ".join(sorted(OBJECTIVES))
            raise ValueError(f"unsupported objective '{self.objective}', expected one of: {supported}")
        if self.shift_ce_eps <= 0.0:
            raise ValueError("shift_ce_eps must be positive")
        if self.objective != "min_fi" and self.model_stem == "ph1_min_fi":
            self.model_stem = f"ph1_{self.objective}"

    @property
    def run_name(self) -> str:
        suffix = period_range_artifact_suffix(self.nqubit, self.period_range)
        base_name = f"{self.model_stem}_{self.nqubit}q{suffix}"
        if self.variant_tag:
            return f"{base_name}_{self.variant_tag}"
        return base_name

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
    min_fi: float | None = None
    mean_shift_l1: float | None = None


@dataclass(slots=True)
class TrainArtifacts:
    history: list[EpochResult]
    final_loss: float
    best_loss: float
    final_min_fi: float | None
    best_epoch: int
    best_min_fi: float | None
    final_mean_shift_l1: float | None
    best_mean_shift_l1: float | None
    model_path: Path
    phase_path: Path
    history_path: Path
    log_path: Path


@dataclass(slots=True)
class ModelCheckpoint:
    epoch: int
    loss: float
    min_fi: float | None
    mean_shift_l1: float | None
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
        "objective": config.objective,
        "exact_support": config.exact_support,
        "variant_tag": config.variant_tag,
        "shift_ce_eps": config.shift_ce_eps,
        "train_device": config.train_device,
    }


def prepare_output_dirs(config: TrainConfig) -> None:
    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)


def resolve_train_device(config: TrainConfig) -> torch.device:
    return torch.device(resolve_compute_device(config.train_device))


def sample_phase_tensor(phase_count: int) -> torch.Tensor:
    return 2 * torch.pi * torch.rand(phase_count, dtype=torch.float32)


def create_model(
    config: TrainConfig,
    init_phases: Sequence[float] | None = None,
) -> PH1MinFIModel:
    return PH1MinFIModel(nqubit=config.nqubit, init_phases=init_phases).to(
        resolve_train_device(config)
    )


def _evaluate_objective(
    model: PH1MinFIModel,
    config: TrainConfig,
) -> tuple[Tensor, float | None, float | None]:
    if config.objective == "min_fi":
        min_fi_value = (
            model.min_fi(
                config.period_range,
                exact_support=True,
            )
            if config.exact_support and hasattr(model, "min_fi")
            else model(config.period_range)
        )
        return -min_fi_value, float(min_fi_value.detach().cpu().item()), None

    if config.objective == "shift_ce_mean":
        loss_value, mean_shift_l1 = model.shift_ce_mean_loss(
            config.period_range,
            exact_support=config.exact_support,
            eps=config.shift_ce_eps,
        )
        return (
            loss_value,
            None,
            float(mean_shift_l1.detach().cpu().item()),
        )

    raise ValueError(f"unsupported objective '{config.objective}'")


def select_monte_carlo_init_phases(config: TrainConfig) -> tuple[list[float], float]:
    if config.monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive for Monte Carlo init")

    candidate_model = create_model(config)
    best_phases: list[float] | None = None
    best_score: float | None = None

    with torch.no_grad():
        for _ in range(config.monte_carlo_samples):
            sampled_phases = sample_phase_tensor(candidate_model.phase_count)
            candidate_model.phases.copy_(sampled_phases.to(candidate_model.phases.device))
            loss_value, min_fi_value, _ = _evaluate_objective(candidate_model, config)
            score = (
                float(min_fi_value)
                if min_fi_value is not None
                else -float(loss_value.detach().cpu().item())
            )

            if best_phases is None or best_score is None or score > best_score:
                best_phases = sampled_phases.detach().cpu().tolist()
                best_score = score

    if best_phases is None:
        raise RuntimeError("monte carlo initialization did not produce any candidate")

    assert best_score is not None
    return best_phases, best_score


def initialize_model(config: TrainConfig, logger: logging.Logger) -> PH1MinFIModel:
    if config.monte_carlo_samples < 1:
        return create_model(config)

    logger.info(
        "running monte carlo initialization samples=%s",
        config.monte_carlo_samples,
    )
    init_phases, init_score = select_monte_carlo_init_phases(config)
    logger.info(
        "selected monte carlo initialization objective=%s score=%.8f",
        config.objective,
        init_score,
    )
    return create_model(config, init_phases=init_phases)


def create_optimizer(model: PH1MinFIModel, config: TrainConfig) -> Optimizer:
    return Adam(model.parameters(), lr=config.learning_rate)


def checkpoint_model(model: PH1MinFIModel, result: EpochResult) -> ModelCheckpoint:
    return ModelCheckpoint(
        epoch=result.epoch,
        loss=result.loss,
        min_fi=result.min_fi,
        mean_shift_l1=result.mean_shift_l1,
        state_dict=snapshot_model_state(model),
        phases=model.export_phases(),
    )


def train_step(
    model: PH1MinFIModel,
    optimizer: Optimizer,
    config: TrainConfig,
) -> tuple[float, float | None, float | None]:
    optimizer.zero_grad()
    loss, _, _ = _evaluate_objective(model, config)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        updated_loss, updated_min_fi, updated_mean_shift_l1 = _evaluate_objective(model, config)

    return (
        float(updated_loss.detach().cpu().item()),
        updated_min_fi,
        updated_mean_shift_l1,
    )


def log_epoch(logger: logging.Logger, result: EpochResult, total_epochs: int) -> None:
    metrics: list[str] = []
    if result.min_fi is not None:
        metrics.append(f"min_fi={result.min_fi:.8f}")
    if result.mean_shift_l1 is not None:
        metrics.append(f"mean_shift_l1={result.mean_shift_l1:.8f}")
    metric_suffix = f" {' '.join(metrics)}" if metrics else ""
    logger.info(
        "epoch=%s/%s loss=%.8f%s",
        result.epoch,
        total_epochs,
        result.loss,
        metric_suffix,
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
        loss_value, min_fi_value, mean_shift_l1 = train_step(model, optimizer, config)
        result = EpochResult(
            epoch=epoch,
            loss=loss_value,
            min_fi=min_fi_value,
            mean_shift_l1=mean_shift_l1,
        )
        history.append(result)
        if best_checkpoint is None or result.loss < best_checkpoint.loss:
            best_checkpoint = checkpoint_model(model, result)

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
        "objective": config.objective,
        "exact_support": config.exact_support,
        "variant_tag": config.variant_tag,
        "best_loss": checkpoint.loss,
        "best_min_fi": checkpoint.min_fi,
        "best_mean_shift_l1": checkpoint.mean_shift_l1,
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

    final_result = history[-1]
    logger.info(
        "training finished objective=%s final_loss=%.8f best_loss=%.8f best_epoch=%s final_min_fi=%s best_min_fi=%s final_mean_shift_l1=%s best_mean_shift_l1=%s",
        config.objective,
        final_result.loss,
        best_checkpoint.loss,
        best_checkpoint.epoch,
        f"{final_result.min_fi:.8f}" if final_result.min_fi is not None else "None",
        f"{best_checkpoint.min_fi:.8f}" if best_checkpoint.min_fi is not None else "None",
        f"{final_result.mean_shift_l1:.8f}" if final_result.mean_shift_l1 is not None else "None",
        f"{best_checkpoint.mean_shift_l1:.8f}" if best_checkpoint.mean_shift_l1 is not None else "None",
    )
    return TrainArtifacts(
        history=history,
        final_loss=final_result.loss,
        best_loss=best_checkpoint.loss,
        final_min_fi=final_result.min_fi,
        best_epoch=best_checkpoint.epoch,
        best_min_fi=best_checkpoint.min_fi,
        final_mean_shift_l1=final_result.mean_shift_l1,
        best_mean_shift_l1=best_checkpoint.mean_shift_l1,
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
    logger.info("ph1 runtime device=%s", resolve_train_device(config))

    model = initialize_model(config, logger)
    optimizer = create_optimizer(model, config)
    history, best_checkpoint = run_training(model, optimizer, config, logger)
    save_model_artifacts(config, best_checkpoint)
    save_history(config, history)
    return summarize_training(config, history, best_checkpoint, logger)
