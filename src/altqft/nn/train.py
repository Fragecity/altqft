from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

from altqft.nn.devices import resolve_compute_device
from altqft.nn.model import ANSATZES, OBJECTIVES, HP1SharedParameterModel, PH1MinFIModel
from altqft.nn.periods import period_range_artifact_suffix
from altqft.nn.runtime import configure_logger, set_random_seed, snapshot_model_state

LOGGER_NAME = "altqft.nn.train"
SerializedConfig = dict[str, int | float | str | bool | list[int] | None]
MODEL_REGISTRY_FILENAME = "README.md"


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
    ansatz: str = "HP1"
    min_fi_weight: float = 1.0
    shift_inv_weight: float | None = None
    shift_inv_period_samples: int = 4
    shift_inv_shift_samples: int = 4
    min_epochs: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1e-6

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
        if self.ansatz not in ANSATZES:
            supported = ", ".join(sorted(ANSATZES))
            raise ValueError(f"unsupported ansatz '{self.ansatz}', expected one of: {supported}")
        if self.shift_ce_eps <= 0.0:
            raise ValueError("shift_ce_eps must be positive")
        if self.min_fi_weight <= 0.0:
            raise ValueError("min_fi_weight must be positive")
        if self.shift_inv_weight is not None and self.shift_inv_weight <= 0.0:
            raise ValueError("shift_inv_weight must be positive")
        if self.shift_inv_period_samples < 1:
            raise ValueError("shift_inv_period_samples must be positive")
        if self.shift_inv_shift_samples < 1:
            raise ValueError("shift_inv_shift_samples must be positive")
        if self.objective == "hp1_shared_fi_shift" and self.ansatz != "HP1_shared":
            raise ValueError("hp1_shared_fi_shift objective requires ansatz='HP1_shared'")
        if self.min_epochs < 0:
            raise ValueError("min_epochs must be non-negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta <= 0.0:
            raise ValueError("early_stopping_min_delta must be positive")
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs must not exceed epochs")
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
        "ansatz": config.ansatz,
        "min_fi_weight": config.min_fi_weight,
        "shift_inv_weight": config.shift_inv_weight,
        "shift_inv_period_samples": config.shift_inv_period_samples,
        "shift_inv_shift_samples": config.shift_inv_shift_samples,
        "min_epochs": config.min_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
    }


def prepare_output_dirs(config: TrainConfig) -> None:
    for path in (config.model_dir, config.data_dir, config.output_dir):
        path.mkdir(parents=True, exist_ok=True)


def training_progress_total(config: TrainConfig) -> int:
    if config.early_stopping_patience > 0:
        return min(
            config.epochs,
            config.min_epochs + config.early_stopping_patience,
        )
    return config.epochs


def render_training_progress(
    epoch: int,
    *,
    total: int,
    best_loss: float | None,
    stale_epochs: int,
    patience: int,
    width: int = 30,
) -> str:
    if total < 1:
        raise ValueError("total must be positive")

    displayed_epoch = min(epoch, total)
    filled = round(width * displayed_epoch / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * displayed_epoch / total
    best_loss_text = "None" if best_loss is None else f"{best_loss:.8g}"
    stale_text = f" stale={stale_epochs}/{patience}" if patience > 0 else ""
    return (
        f"[{bar}] {percent:6.2f}% "
        f"progress={displayed_epoch}/{total} epoch={epoch} "
        f"best_loss={best_loss_text}{stale_text}"
    )


def print_training_progress(
    epoch: int,
    *,
    total: int,
    best_loss: float | None,
    stale_epochs: int,
    patience: int,
) -> None:
    message = render_training_progress(
        epoch,
        total=total,
        best_loss=best_loss,
        stale_epochs=stale_epochs,
        patience=patience,
    )
    print(f"\r{message}", end="", file=sys.stdout, flush=True)


def finish_training_progress() -> None:
    print(file=sys.stdout, flush=True)


def resolve_train_device(config: TrainConfig) -> torch.device:
    return torch.device(resolve_compute_device(config.train_device))


def sample_phase_tensor(phase_count: int) -> torch.Tensor:
    return 2 * torch.pi * torch.rand(phase_count, dtype=torch.float32)


def create_model(
    config: TrainConfig,
    init_phases: Sequence[float] | None = None,
) -> PH1MinFIModel:
    model_cls = HP1SharedParameterModel if config.ansatz == "HP1_shared" else PH1MinFIModel
    return model_cls(nqubit=config.nqubit, init_phases=init_phases).to(resolve_train_device(config))


def _resolved_shift_inv_weight(config: TrainConfig) -> float:
    return 1.0 if config.shift_inv_weight is None else config.shift_inv_weight


def calibrate_shift_inv_weight(
    model: PH1MinFIModel,
    config: TrainConfig,
    logger: logging.Logger,
) -> None:
    if config.objective != "hp1_shared_fi_shift" or config.shift_inv_weight is not None:
        return
    if not isinstance(model, HP1SharedParameterModel):
        raise TypeError("hp1_shared_fi_shift objective requires HP1SharedParameterModel")

    with torch.no_grad():
        min_fi_value = model.min_fi(config.period_range, exact_support=config.exact_support)
        shift_inv_value, _ = model.sampled_shift_invariance_loss(
            period_samples=config.shift_inv_period_samples,
            shift_samples=config.shift_inv_shift_samples,
            exact_support=config.exact_support,
            eps=config.shift_ce_eps,
        )
    min_fi_scalar = abs(float(min_fi_value.detach().cpu().item()))
    shift_inv_scalar = abs(float(shift_inv_value.detach().cpu().item()))
    config.shift_inv_weight = (
        config.min_fi_weight * min_fi_scalar / shift_inv_scalar
        if shift_inv_scalar > 1e-12
        else 1.0
    )
    logger.info(
        "auto calibrated hp1_shared shift_inv_weight=%.8g min_fi=%.8g shift_inv=%.8g",
        config.shift_inv_weight,
        min_fi_scalar,
        shift_inv_scalar,
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

    if config.objective == "hp1_shared_fi_shift":
        if not isinstance(model, HP1SharedParameterModel):
            raise TypeError("hp1_shared_fi_shift objective requires HP1SharedParameterModel")
        min_fi_value = model.min_fi(
            config.period_range,
            exact_support=config.exact_support,
        )
        shift_inv_value, mean_shift_l1 = model.sampled_shift_invariance_loss(
            period_samples=config.shift_inv_period_samples,
            shift_samples=config.shift_inv_shift_samples,
            exact_support=config.exact_support,
            eps=config.shift_ce_eps,
        )
        loss_value = (
            -config.min_fi_weight * min_fi_value
            + _resolved_shift_inv_weight(config) * shift_inv_value
        )
        return (
            loss_value,
            float(min_fi_value.detach().cpu().item()),
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
    best_loss: float | None = None
    stale_epochs = 0
    patience = config.early_stopping_patience
    progress_total = training_progress_total(config)

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
            if best_loss is not None and patience > 0 and epoch > config.min_epochs:
                relative_improvement = (best_loss - result.loss) / max(abs(best_loss), 1e-12)
                if relative_improvement > config.early_stopping_min_delta:
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            best_loss = result.loss
        elif patience > 0 and epoch > config.min_epochs:
            stale_epochs += 1

        if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
            log_epoch(logger, result, config.epochs)

        print_training_progress(
            epoch,
            total=progress_total,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            patience=patience,
        )

        if patience > 0 and epoch > config.min_epochs and stale_epochs >= patience:
            logger.info(
                "early stopping at epoch=%s after %s stale epoch(s); best_loss=%.8g",
                epoch,
                stale_epochs,
                best_loss,
            )
            break

    finish_training_progress()

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
        "ansatz": config.ansatz,
        "min_fi_weight": config.min_fi_weight,
        "shift_inv_weight": config.shift_inv_weight,
        "shift_inv_period_samples": config.shift_inv_period_samples,
        "shift_inv_shift_samples": config.shift_inv_shift_samples,
        "shift_inv_loss": (
            "kl_to_shift_mean"
            if config.objective in {"shift_ce_mean", "hp1_shared_fi_shift"}
            else None
        ),
        "best_epoch": checkpoint.epoch,
        "best_loss": checkpoint.loss,
        "best_min_fi": checkpoint.min_fi,
        "best_mean_shift_l1": checkpoint.mean_shift_l1,
    }
    config.phase_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    refresh_model_registry(config.model_dir)


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _format_optional_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.8g}"
    return str(value)


def _format_period_range(value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        return ""
    if not value:
        return ""
    if value == list(range(value[0], value[-1] + 1)):
        return f"{value[0]}-{value[-1]}"
    if len(value) <= 6:
        return ", ".join(str(item) for item in value)
    return f"{value[0]}, ..., {value[-1]} ({len(value)} values)"


def _phase_payload(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _checkpoint_name_for_phase_path(path: Path) -> str:
    suffix = "_phases.json"
    if path.name.endswith(suffix):
        return f"{path.name[:-len(suffix)]}.pt"
    return path.with_suffix(".pt").name


def _registry_row(path: Path, payload: dict[str, Any]) -> list[str]:
    period_samples = payload.get("shift_inv_period_samples", "")
    shift_samples = payload.get("shift_inv_shift_samples", "")
    shift_sample_text = (
        f"{period_samples}x{shift_samples}"
        if isinstance(period_samples, int) and isinstance(shift_samples, int)
        else ""
    )
    return [
        path.name,
        _checkpoint_name_for_phase_path(path),
        str(payload.get("ansatz", "HP1")),
        str(payload.get("objective", "min_fi")),
        str(payload.get("nqubit", "")),
        _format_period_range(payload.get("period_range")),
        _format_optional_float(payload.get("min_fi_weight", "")),
        _format_optional_float(payload.get("shift_inv_weight", "")),
        str(payload.get("shift_inv_loss", "")),
        shift_sample_text,
        str(payload.get("best_epoch", "")),
        _format_optional_float(payload.get("best_min_fi")),
        _format_optional_float(payload.get("best_loss")),
    ]


def refresh_model_registry(model_dir: Path) -> Path:
    registry_path = model_dir / MODEL_REGISTRY_FILENAME
    phase_paths = sorted(model_dir.glob("*_phases.json"))
    headers = [
        "Parameter file",
        "Checkpoint",
        "Model",
        "Objective",
        "nqubit",
        "Periods",
        "alpha",
        "beta",
        "Shift loss",
        "Shift samples",
        "Best epoch",
        "Best min FI",
        "Best loss",
    ]
    rows: list[list[str]] = []

    for path in phase_paths:
        payload = _phase_payload(path)
        if payload is None:
            continue
        rows.append(_registry_row(path, payload))

    lines = [
        "# Model Registry",
        "",
        "This table maps saved parameter files to their trained model configuration.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(cell) for cell in row) + " |"
        for row in rows
    )
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return registry_path


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
    calibrate_shift_inv_weight(model, config, logger)
    optimizer = create_optimizer(model, config)
    history, best_checkpoint = run_training(model, optimizer, config, logger)
    save_model_artifacts(config, best_checkpoint)
    save_history(config, history)
    return summarize_training(config, history, best_checkpoint, logger)
