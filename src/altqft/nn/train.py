from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam, Optimizer

from altqft.nn.model import PH1MinFIModel


LOGGER_NAME = "altqft.nn.train"


@dataclass(slots=True)
class TrainConfig:
    nqubit: int
    period_range: list[int]
    epochs: int = 100
    learning_rate: float = 0.05
    seed: int = 7
    log_interval: int = 10
    model_dir: Path = Path("model")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model_stem: str = "ph1_min_fi"

    def __post_init__(self) -> None:
        if self.nqubit < 2:
            raise ValueError("nqubit 至少需要为 2。")
        if not self.period_range:
            raise ValueError("period_range 不能为空。")

    @property
    def run_name(self) -> str:
        return f"{self.model_stem}_{self.nqubit}q"

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.run_name}.pt"

    @property
    def phase_path(self) -> Path:
        return self.model_dir / f"{self.run_name}_phases.json"

    @property
    def history_path(self) -> Path:
        return self.data_dir / f"{self.run_name}_history.json"

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
    model_path: Path
    phase_path: Path
    history_path: Path
    log_path: Path


def build_default_period_range(nqubit: int) -> list[int]:
    """默认使用一个较密的 period 区间，避免训练只落在对参数不敏感的周期点上。"""
    dimension = 2**nqubit
    upper_bound = min(dimension - 1, 2 * nqubit)
    periods = list(range(2, upper_bound))
    if not periods:
        raise ValueError(f"nqubit={nqubit} 时无法构造默认的 period_range。")
    return periods


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_output_dirs(config: TrainConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def create_model(config: TrainConfig) -> PH1MinFIModel:
    return PH1MinFIModel(nqubit=config.nqubit)


def create_optimizer(model: PH1MinFIModel, config: TrainConfig) -> Optimizer:
    return Adam(model.parameters(), lr=config.learning_rate)


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
    return float(loss.detach().cpu().item()), float(min_fi_value.detach().cpu().item())


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
) -> list[EpochResult]:
    history: list[EpochResult] = []

    for epoch in range(1, config.epochs + 1):
        loss_value, min_fi_value = train_step(model, optimizer, config.period_range)
        epoch_result = EpochResult(epoch=epoch, loss=loss_value, min_fi=min_fi_value)
        history.append(epoch_result)

        if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
            log_epoch(logger, epoch_result, config.epochs)

    return history


def save_history(config: TrainConfig, history: list[EpochResult]) -> None:
    payload = {
        "config": {
            **asdict(config),
            "model_dir": str(config.model_dir),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
        },
        "history": [asdict(item) for item in history],
    }
    config.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_artifacts(model: PH1MinFIModel, config: TrainConfig) -> None:
    torch.save(model.state_dict(), config.model_path)
    phase_payload = {
        "nqubit": config.nqubit,
        "period_range": config.period_range,
        "phases": model.export_phases(),
    }
    config.phase_path.write_text(json.dumps(phase_payload, indent=2), encoding="utf-8")


def summarize_training(
    config: TrainConfig,
    history: list[EpochResult],
    logger: logging.Logger,
) -> TrainArtifacts:
    if not history:
        raise RuntimeError("训练历史为空，无法生成总结。")

    final_min_fi = history[-1].min_fi
    logger.info("training finished, final_min_fi=%.8f", final_min_fi)

    return TrainArtifacts(
        history=history,
        final_min_fi=final_min_fi,
        model_path=config.model_path,
        phase_path=config.phase_path,
        history_path=config.history_path,
        log_path=config.log_path,
    )


def train_model(config: TrainConfig) -> TrainArtifacts:
    prepare_output_dirs(config)
    logger = configure_logger(config.log_path)
    set_random_seed(config.seed)
    logger.info("start training with config=%s", json.dumps(_serialize_config(config), ensure_ascii=False))

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    history = run_training(model, optimizer, config, logger)
    save_model_artifacts(model, config)
    save_history(config, history)
    return summarize_training(config, history, logger)


def _serialize_config(config: TrainConfig) -> dict[str, Any]:
    return {
        "nqubit": config.nqubit,
        "period_range": config.period_range,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "log_interval": config.log_interval,
        "model_dir": str(config.model_dir),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "model_stem": config.model_stem,
    }
