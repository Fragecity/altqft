from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from altqft.circuits.ph_generators import ph_1_parametrized
from altqft.nn.periods import build_default_period_range
from altqft.nn.train import TrainConfig, train_model


@dataclass(frozen=True, slots=True)
class OptimizedPH1Artifact:
    nqubit: int
    period_range: list[int]
    phases: list[float]
    circuit: QuantumCircuit
    objective: str
    exact_support: bool
    variant_tag: str | None
    model_path: Path
    phase_path: Path
    history_path: Path
    log_path: Path
    reused_existing: bool
    final_min_fi: float | None
    final_loss: float | None
    final_mean_shift_l1: float | None


def load_phase_payload(phase_path: Path) -> dict[str, Any] | None:
    if not phase_path.exists():
        return None

    payload = json.loads(phase_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def phase_artifact_is_current(
    payload: dict[str, Any] | None,
    nqubit: int,
    period_range: list[int],
    *,
    objective: str,
    exact_support: bool,
    variant_tag: str | None,
) -> bool:
    if payload is None:
        return False
    if payload.get("nqubit") != nqubit:
        return False

    phases = payload.get("phases")
    stored_range = payload.get("period_range")
    return (
        isinstance(phases, list)
        and stored_range == period_range
        and payload.get("objective", "min_fi") == objective
        and bool(payload.get("exact_support", False)) == exact_support
        and payload.get("variant_tag") == variant_tag
    )


def _default_model_stem(objective: str) -> str:
    return "ph1_min_fi" if objective == "min_fi" else f"ph1_{objective}"


def build_fi_train_config(
    nqubit: int,
    *,
    period_range: list[int] | None = None,
    epochs: int,
    learning_rate: float,
    seed: int,
    log_interval: int,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    objective: str = "min_fi",
    exact_support: bool = False,
    variant_tag: str | None = None,
    model_stem: str | None = None,
    train_device: str = "auto",
) -> TrainConfig:
    return TrainConfig(
        nqubit=nqubit,
        period_range=list(period_range) if period_range is not None else build_default_period_range(nqubit),
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        log_interval=log_interval,
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        objective=objective,
        exact_support=exact_support,
        variant_tag=variant_tag,
        model_stem=model_stem or _default_model_stem(objective),
        train_device=train_device,
    )


def _artifact_from_payload(
    config: TrainConfig,
    payload: dict[str, Any],
    *,
    reused_existing: bool,
    final_min_fi: float | None,
    final_loss: float | None,
    final_mean_shift_l1: float | None,
) -> OptimizedPH1Artifact:
    phases = payload.get("phases")
    if not isinstance(phases, list) or not all(isinstance(value, (int, float)) for value in phases):
        raise ValueError(f"invalid phase payload in {config.phase_path}")

    phase_values = [float(value) for value in phases]
    return OptimizedPH1Artifact(
        nqubit=config.nqubit,
        period_range=list(config.period_range),
        phases=phase_values,
        circuit=ph_1_parametrized(config.nqubit, phase_values),
        objective=str(payload.get("objective", config.objective)),
        exact_support=bool(payload.get("exact_support", config.exact_support)),
        variant_tag=payload.get("variant_tag"),
        model_path=config.model_path,
        phase_path=config.phase_path,
        history_path=config.history_path,
        log_path=config.log_path,
        reused_existing=reused_existing,
        final_min_fi=final_min_fi,
        final_loss=final_loss,
        final_mean_shift_l1=final_mean_shift_l1,
    )


def ensure_optimized_ph1(
    nqubit: int,
    *,
    period_range: list[int] | None = None,
    epochs: int,
    learning_rate: float = 0.05,
    seed: int = 7,
    log_interval: int = 10,
    model_dir: Path = Path("model"),
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs"),
    force_reoptimize: bool = False,
    require_existing: bool = False,
    objective: str = "min_fi",
    exact_support: bool = False,
    variant_tag: str | None = None,
    model_stem: str | None = None,
    train_device: str = "auto",
) -> OptimizedPH1Artifact:
    config = build_fi_train_config(
        nqubit,
        period_range=period_range,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        log_interval=log_interval,
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        objective=objective,
        exact_support=exact_support,
        variant_tag=variant_tag,
        model_stem=model_stem,
        train_device=train_device,
    )
    payload = load_phase_payload(config.phase_path)

    if not force_reoptimize and phase_artifact_is_current(
        payload,
        config.nqubit,
        config.period_range,
        objective=config.objective,
        exact_support=config.exact_support,
        variant_tag=config.variant_tag,
    ):
        assert payload is not None
        return _artifact_from_payload(
            config,
            payload,
            reused_existing=True,
            final_min_fi=None,
            final_loss=None,
            final_mean_shift_l1=None,
        )

    if require_existing:
        raise FileNotFoundError(
            f"no existing optimized PH1 artifact for nqubit={config.nqubit} period_range={config.period_range} "
            f"objective={config.objective} exact_support={config.exact_support} variant_tag={config.variant_tag} "
            f"at {config.phase_path}"
        )

    train_artifacts = train_model(config)
    refreshed_payload = load_phase_payload(config.phase_path)
    if refreshed_payload is None:
        raise RuntimeError(f"expected phase artifact at {config.phase_path}")

    return _artifact_from_payload(
        config,
        refreshed_payload,
        reused_existing=False,
        final_min_fi=train_artifacts.final_min_fi,
        final_loss=train_artifacts.final_loss,
        final_mean_shift_l1=train_artifacts.final_mean_shift_l1,
    )
