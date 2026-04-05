from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from qiskit.quantum_info import Statevector

from altqft.nn.optimized_ph1 import OptimizedPH1Artifact, ensure_optimized_ph1
from altqft.nn.period_recovery import DeepSetPeriodPredictor
from altqft.nn.periods import build_period_range, period_range_artifact_suffix


@dataclass(slots=True)
class PH1ShorConfig:
    N: int = 15
    a: int = 2
    nqubit: int = 4
    period_min: int = 2
    period_max: int | None = None
    measurement_count: int = 16_384
    top_k: int = 3
    seed: int = 7
    model_dir: Path = Path("model")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    prefer_smoke_artifacts: bool = False
    allow_phase_retraining: bool = True
    variant_tag: str | None = None
    ph1_objective: str = "min_fi"
    exact_support: bool = False

    def __post_init__(self) -> None:
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if self.a <= 1 or self.a >= self.N:
            raise ValueError("a must satisfy 1 < a < N")
        if self.nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if self.measurement_count < 1:
            raise ValueError("measurement_count must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.N > 1 << self.nqubit:
            raise ValueError("N must fit into the chosen nqubit state space")

    @property
    def candidate_periods(self) -> tuple[int, ...]:
        return tuple(
            build_period_range(
                self.nqubit,
                min_period=self.period_min,
                max_period=self.period_max,
            )
        )


@dataclass(frozen=True, slots=True)
class PH1ShorResult:
    success: bool
    factors: tuple[int, int] | None
    predicted_period: int | None
    top1_period: int | None
    measured_work_value: int
    support_x: tuple[int, ...]
    top_periods: tuple[int, ...]
    top_scores: tuple[float, ...]
    phase_path: Path
    model_path: Path
    candidate_periods: tuple[int, ...]
    measurement_count: int


def _artifact_root(model_dir: Path, prefer_smoke_artifacts: bool) -> Path:
    return model_dir / "smoke" if prefer_smoke_artifacts else model_dir


def _period_recovery_run_name(config: PH1ShorConfig) -> str:
    suffix = period_range_artifact_suffix(config.nqubit, config.candidate_periods)
    base_name = f"period_recovery_{config.nqubit}q{suffix}"
    if config.variant_tag:
        return f"{base_name}_{config.variant_tag}"
    return base_name


def _period_model_path(config: PH1ShorConfig) -> Path:
    preferred_root = _artifact_root(config.model_dir, config.prefer_smoke_artifacts)
    preferred = preferred_root / f"{_period_recovery_run_name(config)}.pt"
    if preferred.exists():
        return preferred
    fallback_roots = [config.model_dir, config.model_dir / "smoke"]
    for root in fallback_roots:
        candidate = root / f"{_period_recovery_run_name(config)}.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no period recovery checkpoint found for {config.nqubit} qubits and period_range={list(config.candidate_periods)}"
    )


def _load_period_recovery_model(
    config: PH1ShorConfig,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...], Path]:
    checkpoint_path = _period_model_path(config)
    payload = torch.load(checkpoint_path, map_location="cpu")
    candidate_periods = payload.get("candidate_periods")
    state_dict = payload.get("state_dict")
    if not isinstance(candidate_periods, list) or not all(isinstance(value, int) for value in candidate_periods):
        raise ValueError(f"invalid candidate_periods in {checkpoint_path}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid state_dict in {checkpoint_path}")

    model = DeepSetPeriodPredictor(config.nqubit, len(candidate_periods))
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    model.eval()
    return model, tuple(candidate_periods), checkpoint_path


def _modular_exponentiation_outputs(config: PH1ShorConfig) -> np.ndarray:
    outputs = np.empty(1 << config.nqubit, dtype=np.int64)
    value = 1
    for exponent in range(outputs.size):
        if exponent == 0:
            outputs[exponent] = 1
            continue
        value = (value * config.a) % config.N
        outputs[exponent] = value
    return outputs


def _sample_work_register_value(
    outputs: np.ndarray,
    *,
    seed: int,
) -> int:
    unique_values, counts = np.unique(outputs, return_counts=True)
    probabilities = counts.astype(np.float64) / float(outputs.size)
    rng = np.random.default_rng(seed)
    sampled_index = int(rng.choice(len(unique_values), p=probabilities))
    return int(unique_values[sampled_index])


def _collapsed_periodic_state(
    config: PH1ShorConfig,
    *,
    outputs: np.ndarray,
    work_value: int,
) -> tuple[Statevector, tuple[int, ...]]:
    support = tuple(int(index) for index in np.flatnonzero(outputs == work_value))
    if not support:
        raise ValueError(f"work value {work_value} is not in the modular exponentiation image")

    amplitudes = np.zeros(1 << config.nqubit, dtype=np.complex128)
    amplitudes[list(support)] = 1.0 / math.sqrt(len(support))
    return Statevector(amplitudes, dims=(2,) * config.nqubit), support


def _sample_ph1_bitmatrix(
    config: PH1ShorConfig,
    state: Statevector,
    optimized_ph1: OptimizedPH1Artifact,
) -> np.ndarray:
    ph1_state = state.evolve(optimized_ph1.circuit)
    probabilities = np.asarray(ph1_state.probabilities(), dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(config.seed + 1)
    columns = rng.choice(1 << config.nqubit, size=config.measurement_count, p=probabilities)
    bit_positions = np.arange(config.nqubit - 1, -1, -1, dtype=np.int64)
    return ((columns[:, None] >> bit_positions) & 1).astype(np.int8)


def _recover_factors_from_period(
    period: int,
    *,
    a: int,
    N: int,
) -> tuple[int, int] | None:
    if period % 2 != 0:
        return None
    half_power = pow(a, period // 2, N)
    if half_power in (1, N - 1):
        return None
    left = math.gcd(half_power - 1, N)
    right = math.gcd(half_power + 1, N)
    if 1 < left < N and 1 < right < N:
        return tuple(sorted((left, right)))
    return None


def _select_factorable_period(
    candidate_periods: tuple[int, ...],
    *,
    a: int,
    N: int,
) -> tuple[int | None, tuple[int, int] | None]:
    for candidate_period in candidate_periods:
        factors = _recover_factors_from_period(candidate_period, a=a, N=N)
        if factors is not None:
            return candidate_period, factors
    return None, None


def run_shor_with_ph1(config: PH1ShorConfig) -> PH1ShorResult:
    expected_candidate_periods = config.candidate_periods
    if config.prefer_smoke_artifacts:
        artifact_model_dir = config.model_dir / "smoke"
    else:
        artifact_model_dir = config.model_dir

    optimized_ph1 = ensure_optimized_ph1(
        config.nqubit,
        period_range=list(expected_candidate_periods),
        epochs=1,
        seed=config.seed,
        model_dir=artifact_model_dir,
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        force_reoptimize=False,
        require_existing=not config.allow_phase_retraining,
        objective=config.ph1_objective,
        exact_support=config.exact_support,
        variant_tag=config.variant_tag,
    )
    model, candidate_periods, checkpoint_path = _load_period_recovery_model(config)
    if candidate_periods != expected_candidate_periods:
        raise ValueError(
            f"period model candidate_periods={candidate_periods} do not match expected {expected_candidate_periods}"
        )
    if tuple(optimized_ph1.period_range) != candidate_periods:
        raise ValueError("optimized PH1 period range does not match period recovery checkpoint")

    outputs = _modular_exponentiation_outputs(config)
    measured_work_value = _sample_work_register_value(outputs, seed=config.seed)
    collapsed_state, support_x = _collapsed_periodic_state(
        config,
        outputs=outputs,
        work_value=measured_work_value,
    )
    bitmatrix = _sample_ph1_bitmatrix(config, collapsed_state, optimized_ph1)

    inputs = torch.from_numpy(bitmatrix).unsqueeze(0)
    top_periods_tensor, _, top_scores_tensor = model.predict_topk_periods(
        inputs,
        candidate_periods,
        config.top_k,
    )
    top_periods = tuple(int(value) for value in top_periods_tensor[0].tolist())
    top_scores = tuple(float(value) for value in top_scores_tensor[0].tolist())

    top1_period = top_periods[0] if top_periods else None
    predicted_period, factors = _select_factorable_period(
        top_periods,
        a=config.a,
        N=config.N,
    )
    return PH1ShorResult(
        success=factors is not None,
        factors=factors,
        predicted_period=predicted_period,
        top1_period=top1_period,
        measured_work_value=measured_work_value,
        support_x=support_x,
        top_periods=top_periods,
        top_scores=top_scores,
        phase_path=optimized_ph1.phase_path,
        model_path=checkpoint_path,
        candidate_periods=candidate_periods,
        measurement_count=config.measurement_count,
    )
