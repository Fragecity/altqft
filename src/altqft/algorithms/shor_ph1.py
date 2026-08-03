from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from qiskit.quantum_info import Statevector

from altqft.nn.optimized_ph1 import OptimizedPH1Artifact, ensure_optimized_ph1
from altqft.nn.period_decoder import (
    DeepSetPeriodPredictor,
    predictor_from_checkpoint,
)
from altqft.nn.periods import build_period_range, period_range_artifact_suffix
from altqft.nn.process_qc import _apply_circuit_state


@dataclass(slots=True)
class PH1ShorConfig:
    N: int = 15
    a: int = 2
    nqubit: int = 4
    period_min: int = 2
    period_max: int | None = None
    measurement_count: int = 16_384
    top_k: int = 4
    seed: int = 7
    model_dir: Path = Path("model")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    prefer_smoke_artifacts: bool = False
    allow_phase_retraining: bool = True
    variant_tag: str | None = None
    ph1_objective: str = "min_fi"
    ph1_ansatz: str = "HP1"
    ph1_model_stem: str | None = None
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


def _checkpoint_in_directory(directory: Path) -> Path | None:
    for filename in ("selected.pt", "best.pt"):
        checkpoint = directory / filename
        if checkpoint.exists():
            return checkpoint
    return None


def _matching_distribution_checkpoint(root: Path, prefix: str) -> Path | None:
    for filename in ("selected.pt", "best.pt"):
        matches = sorted(root.glob(f"{prefix}*/{filename}"))
        if matches:
            return matches[-1]
    return None


def _period_model_path(config: PH1ShorConfig) -> Path:
    distribution_stem = (
        f"period_recovery_distribution_{config.nqubit}q"
        f"{period_range_artifact_suffix(config.nqubit, config.candidate_periods)}"
    )
    distribution_prefix = f"{distribution_stem}_"
    roots = (
        _artifact_root(config.model_dir, config.prefer_smoke_artifacts),
        config.model_dir,
        config.model_dir / "smoke",
    )
    for root in dict.fromkeys(roots):
        direct_checkpoint = root / f"{_period_recovery_run_name(config)}.pt"
        if direct_checkpoint.exists():
            return direct_checkpoint

        distribution_directory = root / distribution_stem
        if config.variant_tag:
            distribution_directory = Path(
                f"{distribution_directory}_{config.variant_tag}"
            )
        checkpoint = _checkpoint_in_directory(distribution_directory)
        if checkpoint is not None:
            return checkpoint

        checkpoint = _matching_distribution_checkpoint(root, distribution_prefix)
        if checkpoint is not None:
            return checkpoint
    raise FileNotFoundError(
        f"no period recovery checkpoint found for {config.nqubit} qubits and "
        f"period_range={list(config.candidate_periods)}"
    )


def _load_period_recovery_model(
    config: PH1ShorConfig,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...], Path]:
    checkpoint_path = _period_model_path(config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint in {checkpoint_path}")
    model, candidate_periods = predictor_from_checkpoint(
        payload,
        nqubit=config.nqubit,
    )
    model.eval()
    return model, candidate_periods, checkpoint_path


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
        raise ValueError(
            f"work value {work_value} is not in the modular exponentiation image"
        )

    amplitudes = np.zeros(1 << config.nqubit, dtype=np.complex128)
    amplitudes[list(support)] = 1.0 / math.sqrt(len(support))
    return Statevector(amplitudes, dims=(2,) * config.nqubit), support


def _sample_ph1_bitmatrix(
    config: PH1ShorConfig,
    state: Statevector,
    optimized_ph1: OptimizedPH1Artifact,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.from_numpy(
        np.asarray(state.data, dtype=np.complex64)
    ).to(device=device)
    try:
        ph1_state_tensor = _apply_circuit_state(optimized_ph1.circuit, state_tensor)
        probabilities = ph1_state_tensor.abs().pow(2).cpu().numpy().astype(np.float64)
    except Exception:
        ph1_state = state.evolve(optimized_ph1.circuit)
        probabilities = np.asarray(ph1_state.probabilities(), dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(config.seed + 1)
    columns = rng.choice(
        1 << config.nqubit, size=config.measurement_count, p=probabilities
    )
    bit_positions = np.arange(config.nqubit - 1, -1, -1, dtype=np.int64)
    return ((columns[:, None] >> bit_positions) & 1).astype(np.int8)


def _compress_bitmatrix_counts(bitmatrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patterns, counts = np.unique(bitmatrix, axis=0, return_counts=True)
    weights = counts.astype(np.float32)
    return patterns.astype(np.int8, copy=False), weights


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
        smaller, larger = sorted((left, right))
        return smaller, larger
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
        model_stem=config.ph1_model_stem,
        ansatz=config.ph1_ansatz,
    )
    model, candidate_periods, checkpoint_path = _load_period_recovery_model(config)
    if candidate_periods != expected_candidate_periods:
        raise ValueError(
            f"period model candidate_periods={candidate_periods} do not match expected {expected_candidate_periods}"
        )
    if tuple(optimized_ph1.period_range) != candidate_periods:
        raise ValueError(
            "optimized PH1 period range does not match period recovery checkpoint"
        )

    outputs = _modular_exponentiation_outputs(config)
    measured_work_value = _sample_work_register_value(outputs, seed=config.seed)
    collapsed_state, support_x = _collapsed_periodic_state(
        config,
        outputs=outputs,
        work_value=measured_work_value,
    )
    bitmatrix = _sample_ph1_bitmatrix(config, collapsed_state, optimized_ph1)

    compressed_bits, sample_weights = _compress_bitmatrix_counts(bitmatrix)
    inputs = torch.from_numpy(compressed_bits).unsqueeze(0)
    weights = torch.from_numpy(sample_weights).unsqueeze(0)
    with torch.inference_mode():
        try:
            top_periods_tensor, _, top_scores_tensor = model.predict_topk_periods(
                inputs,
                candidate_periods,
                config.top_k,
                sample_weights=weights,
            )
        except TypeError as exc:
            if "sample_weights" not in str(exc):
                raise
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
