from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from qiskit.quantum_info import Operator

from altqft.nn.optimized_ph1 import OptimizedPH1Artifact, ensure_optimized_ph1
from altqft.nn.period_recovery import (
    DeepSetPeriodPredictor,
    PeriodRecoveryDatasetArtifacts,
    PeriodRecoveryDatasetConfig,
    PeriodRecoveryTrainArtifacts,
    PeriodRecoveryTrainConfig,
    _columns_to_bit_matrix,
    _sample_bit_matrix_from_distribution,
    generate_period_recovery_dataset,
    load_shift_pool_manifest,
    resolve_train_device,
    train_period_recovery,
)
from altqft.nn.periods import build_legacy_period_range
from altqft.nn.process_qc import (
    _exact_support_indices,
    _probability_vector_from_support,
    _surrogate_support_indices,
)
from altqft.nn.runtime import configure_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_VARIANT_TAG = "exact_shiftce_pool10_hold1"
DEFAULT_EXPERIMENT_LOG_NAME = "altqft.experiments.period_noise_9q10q"
DEFAULT_PLOT_STEM = "period_noise_9q10q_accuracy_vs_noise"
DEFAULT_RESULTS_STEM = "period_noise_9q10q_results"
PLOT_COLORS = {
    9: "#2563eb",
    10: "#db2777",
}


@dataclass(frozen=True, slots=True)
class ExperimentRoots:
    model_dir: Path = Path("model/noise_9q10q")
    output_dir: Path = Path("outputs/noise_9q10q")
    dataset_dir: Path = Path("data/period_recovery_noise_9q10q")
    data_dir: Path = Path("data")

    def ensure_dirs(self) -> None:
        for path in (self.model_dir, self.output_dir, self.dataset_dir, self.data_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class PeriodNoiseRecipe:
    variant_tag: str = DEFAULT_VARIANT_TAG
    exact_support: bool = True
    ph1_objective: str = "shift_ce_mean"
    ph1_epochs: int = 200
    ph1_learning_rate: float = 0.05
    ph1_log_interval: int = 10
    ph1_train_device: str = "cuda"
    top_k: int = 10
    batch_size: int = 8
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    label_smoothing: float = 0.05
    min_epochs: int = 15
    early_stopping_patience: int = 15
    seed: int = 7
    log_interval: int = 5
    dataset_mode: str = "shift_pool"
    pool_multiplier: int = 10
    held_out_shifts_per_period: int = 1
    val_draws_per_heldout_shift: int = 4
    train_draws_per_epoch: int = 51200
    cache_device: str = "cuda"


@dataclass(frozen=True, slots=True)
class QubitExperimentSpec:
    nqubit: int
    measurement_count: int
    period_min: int
    period_max: int


@dataclass(frozen=True, slots=True)
class NoiseSweepPoint:
    noise_strength: float
    accuracy: float
    correct: int
    total: int


@dataclass(frozen=True, slots=True)
class QubitNoiseSweepResult:
    nqubit: int
    measurement_count: int
    period_min: int
    period_max: int
    selected_epoch: int
    selected_val_top1: float
    selected_val_topk: float
    phase_path: Path
    model_path: Path
    manifest_path: Path
    points: tuple[NoiseSweepPoint, ...]


@dataclass(frozen=True, slots=True)
class PeriodNoiseRunSummary:
    roots: ExperimentRoots
    recipe: PeriodNoiseRecipe
    noise_levels: tuple[float, ...]
    results: tuple[QubitNoiseSweepResult, ...]
    json_path: Path
    csv_path: Path
    png_path: Path
    svg_path: Path
    log_path: Path


@dataclass(frozen=True, slots=True)
class TrainedQubitExperiment:
    spec: QubitExperimentSpec
    optimized_ph1: OptimizedPH1Artifact
    dataset_artifacts: PeriodRecoveryDatasetArtifacts
    train_artifacts: PeriodRecoveryTrainArtifacts


DEFAULT_EXPERIMENT_ROOTS = ExperimentRoots()
DEFAULT_PERIOD_NOISE_RECIPE = PeriodNoiseRecipe()
DEFAULT_NOISE_LEVELS = tuple(float(value) for value in np.geomspace(1e-2, 1e-3, 10))


def build_noise_levels(
    *,
    start: float = 1e-2,
    stop: float = 1e-3,
    count: int = 10,
) -> tuple[float, ...]:
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("noise sweep bounds must be positive")
    if count < 1:
        raise ValueError("noise sweep count must be positive")
    levels = np.geomspace(start, stop, count)
    if levels[0] < levels[-1]:
        levels = levels[::-1]
    return tuple(float(value) for value in levels.tolist())


def legacy_qubit_experiment_spec(nqubit: int) -> QubitExperimentSpec:
    period_range = build_legacy_period_range(nqubit)
    return QubitExperimentSpec(
        nqubit=nqubit,
        measurement_count=1024 * nqubit**2,
        period_min=int(period_range[0]),
        period_max=int(period_range[-1]),
    )


def apply_global_depolarizing_noise(
    probabilities: np.ndarray,
    noise_strength: float,
) -> np.ndarray:
    if not 0.0 <= noise_strength <= 1.0:
        raise ValueError("noise_strength must be in [0, 1]")
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a vector")

    normalized = np.asarray(probabilities, dtype=np.float64)
    total = float(normalized.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    normalized = np.clip(normalized / total, 0.0, None)
    normalized /= normalized.sum()

    uniform = np.full(normalized.shape, 1.0 / float(normalized.size), dtype=np.float64)
    noisy = (1.0 - noise_strength) * normalized + noise_strength * uniform
    noisy = np.clip(noisy, 0.0, None)
    noisy /= noisy.sum()
    return noisy


def _period_shift_distribution(
    unitary: np.ndarray,
    period: int,
    shift: int,
    *,
    exact_support: bool,
) -> np.ndarray:
    support_indices = (
        _exact_support_indices(unitary.shape[1], period, shift)
        if exact_support
        else _surrogate_support_indices(unitary.shape[1], period, shift)
    )
    distribution = np.asarray(
        _probability_vector_from_support(unitary, support_indices),
        dtype=np.float64,
    )
    distribution = np.clip(distribution, 0.0, None)
    distribution /= distribution.sum()
    distribution[-1] = max(0.0, 1.0 - distribution[:-1].sum())
    return np.asarray(distribution / distribution.sum(), dtype=np.float64)


def _result_paths(roots: ExperimentRoots) -> tuple[Path, Path, Path, Path, Path]:
    json_path = roots.output_dir / f"{DEFAULT_RESULTS_STEM}.json"
    csv_path = roots.output_dir / f"{DEFAULT_RESULTS_STEM}.csv"
    png_path = roots.output_dir / f"{DEFAULT_PLOT_STEM}.png"
    svg_path = roots.output_dir / f"{DEFAULT_PLOT_STEM}.svg"
    log_path = roots.output_dir / "period_noise_9q10q.log"
    return json_path, csv_path, png_path, svg_path, log_path


def _build_dataset_config(
    spec: QubitExperimentSpec,
    roots: ExperimentRoots,
    recipe: PeriodNoiseRecipe,
) -> PeriodRecoveryDatasetConfig:
    return PeriodRecoveryDatasetConfig(
        nqubit=spec.nqubit,
        measurement_count=spec.measurement_count,
        period_min=spec.period_min,
        period_max=spec.period_max,
        seed=recipe.seed,
        dataset_dir=roots.dataset_dir,
        exact_support=recipe.exact_support,
        cache_mode=recipe.dataset_mode,
        pool_multiplier=recipe.pool_multiplier,
        held_out_shifts_per_period=recipe.held_out_shifts_per_period,
        val_draws_per_heldout_shift=recipe.val_draws_per_heldout_shift,
        train_draws_per_epoch=recipe.train_draws_per_epoch,
        variant_tag=recipe.variant_tag,
        cache_device=recipe.cache_device,
    )


def _build_train_config(
    spec: QubitExperimentSpec,
    roots: ExperimentRoots,
    recipe: PeriodNoiseRecipe,
) -> PeriodRecoveryTrainConfig:
    return PeriodRecoveryTrainConfig(
        nqubit=spec.nqubit,
        period_min=spec.period_min,
        period_max=spec.period_max,
        top_k=recipe.top_k,
        batch_size=recipe.batch_size,
        epochs=recipe.epochs,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        dropout=recipe.dropout,
        label_smoothing=recipe.label_smoothing,
        min_epochs=recipe.min_epochs,
        early_stopping_patience=recipe.early_stopping_patience,
        seed=recipe.seed,
        log_interval=recipe.log_interval,
        model_dir=roots.model_dir,
        data_dir=roots.data_dir,
        output_dir=roots.output_dir,
        force_reoptimize_phases=False,
        regenerate_dataset=False,
        fi_epochs=1,
        fi_learning_rate=recipe.ph1_learning_rate,
        fi_log_interval=recipe.ph1_log_interval,
        fi_objective=recipe.ph1_objective,
        fi_exact_support=recipe.exact_support,
        fi_train_device=recipe.ph1_train_device,
        dataset_mode=recipe.dataset_mode,
        variant_tag=recipe.variant_tag,
    )


def _train_qubit_experiment(
    spec: QubitExperimentSpec,
    roots: ExperimentRoots,
    recipe: PeriodNoiseRecipe,
    logger: logging.Logger,
) -> TrainedQubitExperiment:
    dataset_config = _build_dataset_config(spec, roots, recipe)
    train_config = _build_train_config(spec, roots, recipe)
    candidate_periods = list(range(spec.period_min, spec.period_max + 1))
    logger.info(
        "prepare %sq training measurement_count=%s period_range=%s..%s variant_tag=%s",
        spec.nqubit,
        spec.measurement_count,
        spec.period_min,
        spec.period_max,
        recipe.variant_tag,
    )
    optimized_ph1 = ensure_optimized_ph1(
        spec.nqubit,
        period_range=candidate_periods,
        epochs=recipe.ph1_epochs,
        learning_rate=recipe.ph1_learning_rate,
        seed=recipe.seed,
        log_interval=recipe.ph1_log_interval,
        model_dir=roots.model_dir,
        data_dir=roots.data_dir,
        output_dir=roots.output_dir,
        force_reoptimize=False,
        objective=recipe.ph1_objective,
        exact_support=recipe.exact_support,
        variant_tag=recipe.variant_tag,
        train_device=recipe.ph1_train_device,
    )
    dataset_artifacts = generate_period_recovery_dataset(
        dataset_config,
        optimized_ph1,
        regenerate=False,
    )
    train_artifacts = train_period_recovery(train_config, dataset_artifacts, optimized_ph1)
    logger.info(
        "finished %sq training selected_epoch=%s selected_val_top1=%.4f model_path=%s",
        spec.nqubit,
        train_artifacts.selected_epoch,
        train_artifacts.selected_val_top1,
        train_artifacts.model_path,
    )
    return TrainedQubitExperiment(
        spec=spec,
        optimized_ph1=optimized_ph1,
        dataset_artifacts=dataset_artifacts,
        train_artifacts=train_artifacts,
    )


def _load_period_predictor(
    model_path: Path,
    *,
    nqubit: int,
    device: torch.device,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...]]:
    payload = torch.load(model_path, map_location="cpu")
    candidate_periods = payload.get("candidate_periods")
    state_dict = payload.get("state_dict")
    if not isinstance(candidate_periods, list) or not all(isinstance(value, int) for value in candidate_periods):
        raise ValueError(f"invalid candidate_periods in {model_path}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid state_dict in {model_path}")

    model = DeepSetPeriodPredictor(nqubit, len(candidate_periods))
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    model.eval()
    model.to(device)
    return model, tuple(int(value) for value in candidate_periods)


def _val_entry_row_indices(
    *,
    manifest_seed: int,
    period: int,
    shift: int,
    draw_index: int,
    entry_index: int,
    pool_size: int,
    measurement_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        manifest_seed
        + period * 100_003
        + shift * 1_009
        + draw_index * 17
        + entry_index
    )
    return np.asarray(
        rng.choice(pool_size, size=measurement_count, replace=False),
        dtype=np.int64,
    )


def _noise_pool_seed(
    *,
    seed: int,
    noise_index: int,
    period: int,
    shift: int,
) -> int:
    return seed + (noise_index + 1) * 1_000_003 + period * 10_007 + shift * 101


def _evaluate_noise_sweep(
    experiment: TrainedQubitExperiment,
    noise_levels: tuple[float, ...],
    logger: logging.Logger,
) -> QubitNoiseSweepResult:
    manifest_path = experiment.dataset_artifacts.manifest_path or experiment.dataset_artifacts.train_path
    manifest = load_shift_pool_manifest(manifest_path)
    if manifest.candidate_periods != tuple(range(experiment.spec.period_min, experiment.spec.period_max + 1)):
        raise ValueError("manifest candidate periods do not match experiment spec")

    device = resolve_train_device()
    model, candidate_periods = _load_period_predictor(
        experiment.train_artifacts.model_path,
        nqubit=experiment.spec.nqubit,
        device=device,
    )
    if candidate_periods != manifest.candidate_periods:
        raise ValueError("checkpoint candidate periods do not match manifest")

    unitary = np.asarray(Operator(experiment.optimized_ph1.circuit).data, dtype=np.complex128)
    basis_bit_rows = _columns_to_bit_matrix(
        np.arange(1 << experiment.spec.nqubit, dtype=np.int64),
        experiment.spec.nqubit,
    )
    total_examples = len(manifest.candidate_periods) * manifest.val_draws_per_heldout_shift
    non_blocking = device.type == "cuda"
    results: list[NoiseSweepPoint] = []

    logger.info(
        "evaluate %sq noise sweep held_out_examples=%s noise_levels=%s",
        experiment.spec.nqubit,
        total_examples,
        [f"{value:.6f}" for value in noise_levels],
    )
    with torch.inference_mode():
        for noise_index, noise_strength in enumerate(noise_levels):
            correct = 0
            entry_index = 0
            for shard in manifest.shards:
                for shift in shard.val_shifts:
                    ideal_distribution = _period_shift_distribution(
                        unitary,
                        shard.period,
                        shift,
                        exact_support=manifest.exact_support,
                    )
                    noisy_distribution = apply_global_depolarizing_noise(
                        ideal_distribution,
                        noise_strength,
                    )
                    pool_rng = np.random.default_rng(
                        _noise_pool_seed(
                            seed=manifest.seed,
                            noise_index=noise_index,
                            period=shard.period,
                            shift=shift,
                        )
                    )
                    bit_pool = _sample_bit_matrix_from_distribution(
                        noisy_distribution,
                        manifest.pool_size,
                        basis_bit_rows,
                        pool_rng,
                    )

                    batch_matrices = np.empty(
                        (
                            manifest.val_draws_per_heldout_shift,
                            manifest.measurement_count,
                            manifest.nqubit,
                        ),
                        dtype=np.int8,
                    )
                    labels = np.full(
                        manifest.val_draws_per_heldout_shift,
                        shard.label,
                        dtype=np.int64,
                    )
                    for draw_index in range(manifest.val_draws_per_heldout_shift):
                        row_indices = _val_entry_row_indices(
                            manifest_seed=manifest.seed,
                            period=shard.period,
                            shift=shift,
                            draw_index=draw_index,
                            entry_index=entry_index,
                            pool_size=manifest.pool_size,
                            measurement_count=manifest.measurement_count,
                        )
                        batch_matrices[draw_index] = bit_pool[row_indices]
                        entry_index += 1

                    batch = torch.from_numpy(batch_matrices).to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=non_blocking,
                    )
                    predicted = model(batch).argmax(dim=1).detach().cpu().numpy()
                    correct += int((predicted == labels).sum())

            accuracy = correct / float(total_examples)
            logger.info(
                "noise=%0.6f nqubit=%s accuracy=%.4f correct=%s total=%s",
                noise_strength,
                experiment.spec.nqubit,
                accuracy,
                correct,
                total_examples,
            )
            results.append(
                NoiseSweepPoint(
                    noise_strength=float(noise_strength),
                    accuracy=float(accuracy),
                    correct=correct,
                    total=total_examples,
                )
            )

    return QubitNoiseSweepResult(
        nqubit=experiment.spec.nqubit,
        measurement_count=experiment.spec.measurement_count,
        period_min=experiment.spec.period_min,
        period_max=experiment.spec.period_max,
        selected_epoch=experiment.train_artifacts.selected_epoch,
        selected_val_top1=experiment.train_artifacts.selected_val_top1,
        selected_val_topk=experiment.train_artifacts.selected_val_topk,
        phase_path=experiment.optimized_ph1.phase_path,
        model_path=experiment.train_artifacts.model_path,
        manifest_path=manifest_path,
        points=tuple(results),
    )


def _plot_noise_sweep(
    results: tuple[QubitNoiseSweepResult, ...],
    *,
    png_path: Path,
    svg_path: Path,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for result in results:
        noise_levels = [point.noise_strength for point in result.points]
        accuracies = [point.accuracy for point in result.points]
        ax.plot(
            noise_levels,
            accuracies,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=PLOT_COLORS.get(result.nqubit, "#111827"),
            label=f"{result.nqubit}q",
        )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Global depolarizing noise strength")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Post-PH1 Global Depolarizing Noise Sweep")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)


def _summary_payload(summary: PeriodNoiseRunSummary) -> dict[str, Any]:
    return {
        "roots": {key: str(value) for key, value in asdict(summary.roots).items()},
        "recipe": asdict(summary.recipe),
        "noise_levels": list(summary.noise_levels),
        "results": [
            {
                "nqubit": result.nqubit,
                "measurement_count": result.measurement_count,
                "period_min": result.period_min,
                "period_max": result.period_max,
                "selected_epoch": result.selected_epoch,
                "selected_val_top1": result.selected_val_top1,
                "selected_val_topk": result.selected_val_topk,
                "phase_path": str(result.phase_path),
                "model_path": str(result.model_path),
                "manifest_path": str(result.manifest_path),
                "points": [asdict(point) for point in result.points],
            }
            for result in summary.results
        ],
        "json_path": str(summary.json_path),
        "csv_path": str(summary.csv_path),
        "png_path": str(summary.png_path),
        "svg_path": str(summary.svg_path),
        "log_path": str(summary.log_path),
    }


def _write_summary_json(summary: PeriodNoiseRunSummary) -> None:
    summary.json_path.write_text(
        json.dumps(_summary_payload(summary), indent=2),
        encoding="utf-8",
    )


def _write_summary_csv(summary: PeriodNoiseRunSummary) -> None:
    with summary.csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nqubit",
                "measurement_count",
                "period_min",
                "period_max",
                "noise_strength",
                "accuracy",
                "correct",
                "total",
                "selected_epoch",
                "selected_val_top1",
                "selected_val_topk",
                "phase_path",
                "model_path",
                "manifest_path",
            ],
        )
        writer.writeheader()
        for result in summary.results:
            for point in result.points:
                writer.writerow(
                    {
                        "nqubit": result.nqubit,
                        "measurement_count": result.measurement_count,
                        "period_min": result.period_min,
                        "period_max": result.period_max,
                        "noise_strength": point.noise_strength,
                        "accuracy": point.accuracy,
                        "correct": point.correct,
                        "total": point.total,
                        "selected_epoch": result.selected_epoch,
                        "selected_val_top1": result.selected_val_top1,
                        "selected_val_topk": result.selected_val_topk,
                        "phase_path": result.phase_path,
                        "model_path": result.model_path,
                        "manifest_path": result.manifest_path,
                    }
                )


def run_period_noise_experiment(
    *,
    qubit_specs: tuple[QubitExperimentSpec, ...] = (
        legacy_qubit_experiment_spec(9),
        legacy_qubit_experiment_spec(10),
    ),
    roots: ExperimentRoots = DEFAULT_EXPERIMENT_ROOTS,
    recipe: PeriodNoiseRecipe = DEFAULT_PERIOD_NOISE_RECIPE,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
) -> PeriodNoiseRunSummary:
    roots.ensure_dirs()
    json_path, csv_path, png_path, svg_path, log_path = _result_paths(roots)
    logger = configure_logger(DEFAULT_EXPERIMENT_LOG_NAME, log_path)
    logger.info(
        "start period-noise experiment qubits=%s roots=%s",
        [spec.nqubit for spec in qubit_specs],
        {key: str(value) for key, value in asdict(roots).items()},
    )
    logger.info(
        "recipe=%s",
        json.dumps(asdict(recipe)),
    )
    logger.info(
        "noise_levels=%s",
        [f"{value:.6f}" for value in noise_levels],
    )

    experiments = tuple(
        _train_qubit_experiment(spec, roots, recipe, logger)
        for spec in qubit_specs
    )
    results = tuple(
        _evaluate_noise_sweep(experiment, noise_levels, logger)
        for experiment in experiments
    )

    summary = PeriodNoiseRunSummary(
        roots=roots,
        recipe=recipe,
        noise_levels=noise_levels,
        results=results,
        json_path=json_path,
        csv_path=csv_path,
        png_path=png_path,
        svg_path=svg_path,
        log_path=log_path,
    )
    _write_summary_json(summary)
    _write_summary_csv(summary)
    _plot_noise_sweep(results, png_path=png_path, svg_path=svg_path)
    logger.info(
        "finished period-noise experiment json=%s csv=%s png=%s svg=%s",
        json_path,
        csv_path,
        png_path,
        svg_path,
    )
    return summary
