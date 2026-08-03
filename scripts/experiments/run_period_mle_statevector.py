from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector

from altqft.circuits.HPgenerators import HP1_shared_parameter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_NQUBITS = (8, 9, 10, 11)
DEFAULT_MODEL_DIR = Path("model")
DEFAULT_OUTPUT_DIR = Path("outputs/period_mle_8q9q10q11q_statevector")
DEFAULT_SAMPLE_COUNTS = (1, *range(5, 105, 5), *range(125, 2001, 25))
DEFAULT_TRIALS_PER_PERIOD = 200
DEFAULT_TRIAL_BATCH_SIZE = 20
DEFAULT_SEED = 7
PLOT_X_MIN = 1
PLOT_X_MAX = 1750

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SuccessPoint:
    sample_count: int
    success_rate: float
    failure_rate: float
    correct: int
    failed: int
    total: int
    ci95_low: float
    ci95_high: float
    failure_ci95_low: float
    failure_ci95_high: float


@dataclass(frozen=True, slots=True)
class SuccessSeries:
    nqubit: int
    periods: tuple[int, ...]
    period_min: int
    period_max: int
    candidate_count: int
    checkpoint_path: str
    checkpoint_sha256: str
    phase_path: str
    phase_sha256: str
    probability_path: str
    shift: int
    exact_support: bool
    seed: int
    points: tuple[SuccessPoint, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact HP1 output probabilities from statevectors for multiple "
            "qubit counts and compare maximum-likelihood period recovery versus "
            "measurement sample count."
        ),
    )
    parser.add_argument(
        "--nqubits",
        type=int,
        nargs="+",
        default=list(DEFAULT_NQUBITS),
        help="Qubit counts to include as separate curves.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained HP1_shared checkpoints and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for probability, result, and plot artifacts.",
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_COUNTS),
        help="Measurement sample counts used by the MLE.",
    )
    parser.add_argument(
        "--trials-per-period",
        type=int,
        default=DEFAULT_TRIALS_PER_PERIOD,
        help="Independent MLE trials for every true candidate period.",
    )
    parser.add_argument(
        "--trial-batch-size",
        type=int,
        default=DEFAULT_TRIAL_BATCH_SIZE,
        help="Trial batch size used while accumulating likelihoods.",
    )
    parser.add_argument(
        "--period-min",
        type=int,
        default=None,
        help="Optional lower bound applied to the checkpoint candidate periods.",
    )
    parser.add_argument(
        "--period-max",
        type=int,
        default=None,
        help="Optional upper bound applied to the checkpoint candidate periods.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="Coset shift used to construct every periodic input state.",
    )
    parser.add_argument(
        "--exact-support",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use all in-range shift + q*r indices. By default, reuse the support "
            "convention stored in the phase metadata."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def resolve_artifact_paths(model_dir: Path, nqubit: int) -> tuple[Path, Path]:
    phase_paths = sorted(
        model_dir.glob(f"ph1_hp1_shared_fi_shift_{nqubit}q_p*_phases.json")
    )
    if not phase_paths:
        raise FileNotFoundError(
            f"no HP1_shared FI+shift phase metadata found for n={nqubit} in {model_dir}"
        )
    if len(phase_paths) > 1:
        joined = ", ".join(str(path) for path in phase_paths)
        raise RuntimeError(f"multiple phase metadata files found for n={nqubit}: {joined}")

    phase_path = phase_paths[0]
    suffix = "_phases.json"
    checkpoint_path = phase_path.with_name(
        f"{phase_path.name[:-len(suffix)]}.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    return checkpoint_path, phase_path


def load_checkpoint_phases(path: Path) -> list[float]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"expected a state_dict in {path}")
    phases = state_dict.get("phases")
    if not isinstance(phases, torch.Tensor) or phases.ndim != 1:
        raise TypeError(f"checkpoint does not contain a one-dimensional phases tensor: {path}")
    return [float(value) for value in phases.detach().cpu().tolist()]


def metadata_phases(payload: dict[str, Any], path: Path) -> list[float]:
    values = payload.get("phases")
    if not isinstance(values, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        raise TypeError(f"invalid phases in {path}")
    return [float(value) for value in values]


def resolve_periods(
    payload: dict[str, Any],
    *,
    period_min: int | None,
    period_max: int | None,
) -> tuple[int, ...]:
    stored_periods = payload.get("period_range")
    if not isinstance(stored_periods, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in stored_periods
    ):
        raise TypeError("phase metadata does not contain an integer period_range")

    periods = tuple(
        int(period)
        for period in stored_periods
        if (period_min is None or period >= period_min)
        and (period_max is None or period <= period_max)
    )
    if not periods:
        raise ValueError("period filters removed every candidate period")
    if len(set(periods)) != len(periods):
        raise ValueError("candidate periods must be unique")
    return periods


def resolve_sample_counts(values: list[int]) -> tuple[int, ...]:
    if not values or any(value < 1 for value in values):
        raise ValueError("sample counts must be positive")
    return tuple(sorted(set(int(value) for value in values)))


def periodic_input_state(
    *,
    nqubit: int,
    period: int,
    shift: int,
    exact_support: bool,
) -> Statevector:
    size = 1 << nqubit
    if period < 1:
        raise ValueError("period must be positive")
    if not 0 <= shift < period:
        raise ValueError(f"shift={shift} must satisfy 0 <= shift < period={period}")

    if exact_support:
        support = np.arange(shift, size, period, dtype=np.int64)
    else:
        support = shift + np.arange(size // period, dtype=np.int64) * period
    if support.size < 1:
        raise ValueError(f"empty periodic support for period={period}, shift={shift}")

    amplitudes = np.zeros(size, dtype=np.complex128)
    amplitudes[support] = 1.0 / math.sqrt(float(support.size))
    return Statevector(amplitudes, dims=(2,) * nqubit)


def exact_probability_table(
    *,
    nqubit: int,
    phases: list[float],
    periods: tuple[int, ...],
    shift: int,
    exact_support: bool,
) -> FloatArray:
    circuit = HP1_shared_parameter(nqubit, phases)
    probabilities = np.empty((len(periods), 1 << nqubit), dtype=np.float64)

    for row, period in enumerate(periods):
        input_state = periodic_input_state(
            nqubit=nqubit,
            period=period,
            shift=shift,
            exact_support=exact_support,
        )
        output_state = input_state.evolve(circuit)
        distribution = np.asarray(output_state.probabilities(), dtype=np.float64)
        distribution = np.clip(distribution, 0.0, None)
        distribution /= distribution.sum()
        probabilities[row] = distribution

        if row == 0 or (row + 1) % 10 == 0 or row + 1 == len(periods):
            print(
                f"statevector probabilities {row + 1}/{len(periods)} "
                f"period={period}",
                flush=True,
            )

    return probabilities


def log_probability_table(probabilities: FloatArray) -> FloatArray:
    logs = np.full(probabilities.shape, -np.inf, dtype=np.float64)
    np.log(probabilities, out=logs, where=probabilities > 0.0)
    return logs


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1:
        raise ValueError("total must be positive")
    estimate = correct / float(total)
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_mle_success(
    probabilities: FloatArray,
    *,
    sample_counts: tuple[int, ...],
    trials_per_period: int,
    trial_batch_size: int,
    seed: int,
) -> list[SuccessPoint]:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (periods, outcomes)")
    if trials_per_period < 1:
        raise ValueError("trials_per_period must be positive")
    if trial_batch_size < 1:
        raise ValueError("trial_batch_size must be positive")

    rng = np.random.default_rng(seed)
    log_probabilities = log_probability_table(probabilities)
    max_samples = sample_counts[-1]
    correct = np.zeros(len(sample_counts), dtype=np.int64)

    for true_index, true_distribution in enumerate(probabilities):
        completed = 0
        while completed < trials_per_period:
            batch_size = min(trial_batch_size, trials_per_period - completed)
            observations = rng.choice(
                probabilities.shape[1],
                size=(batch_size, max_samples),
                replace=True,
                p=true_distribution,
            )
            observed_log_probabilities = log_probabilities[:, observations]
            cumulative_log_likelihood = np.cumsum(
                observed_log_probabilities,
                axis=-1,
                dtype=np.float64,
            )

            for point_index, sample_count in enumerate(sample_counts):
                scores = cumulative_log_likelihood[:, :, sample_count - 1]
                predicted_indices = scores.argmax(axis=0)
                correct[point_index] += int(
                    np.count_nonzero(predicted_indices == true_index)
                )
            completed += batch_size

        if (
            true_index == 0
            or (true_index + 1) % 10 == 0
            or true_index + 1 == probabilities.shape[0]
        ):
            print(
                f"MLE trials {true_index + 1}/{probabilities.shape[0]}",
                flush=True,
            )

    total = probabilities.shape[0] * trials_per_period
    points: list[SuccessPoint] = []
    for index, sample_count in enumerate(sample_counts):
        point_correct = int(correct[index])
        point_failed = total - point_correct
        success_rate = point_correct / float(total)
        ci95_low, ci95_high = wilson_interval(point_correct, total)
        points.append(
            SuccessPoint(
                sample_count=sample_count,
                success_rate=success_rate,
                failure_rate=1.0 - success_rate,
                correct=point_correct,
                failed=point_failed,
                total=total,
                ci95_low=ci95_low,
                ci95_high=ci95_high,
                failure_ci95_low=1.0 - ci95_high,
                failure_ci95_high=1.0 - ci95_low,
            )
        )
    return points


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_probability_table(
    output_dir: Path,
    *,
    nqubit: int,
    periods: tuple[int, ...],
    probabilities: FloatArray,
    shift: int,
    exact_support: bool,
) -> Path:
    path = output_dir / f"period_mle_{nqubit}q_exact_probabilities.npz"
    np.savez_compressed(
        path,
        periods=np.asarray(periods, dtype=np.int64),
        probabilities=probabilities,
        shift=np.asarray(shift, dtype=np.int64),
        exact_support=np.asarray(exact_support, dtype=np.bool_),
    )
    return path


def save_results(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    series: list[SuccessSeries],
) -> tuple[Path, Path]:
    json_path = output_dir / "period_mle_8q9q10q11q_failure_vs_samples.json"
    csv_path = output_dir / "period_mle_8q9q10q11q_failure_vs_samples.csv"
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "series": [asdict(item) for item in series],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    point_fields = list(asdict(series[0].points[0]))
    fieldnames = [
        "nqubit",
        "period_min",
        "period_max",
        "candidate_count",
        *point_fields,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in series:
            for point in item.points:
                writer.writerow(
                    {
                        "nqubit": item.nqubit,
                        "period_min": item.period_min,
                        "period_max": item.period_max,
                        "candidate_count": item.candidate_count,
                        **asdict(point),
                    }
                )
    return json_path, csv_path


def save_plot(
    output_dir: Path,
    *,
    series: list[SuccessSeries],
) -> tuple[Path, Path, Path]:
    png_path = output_dir / "period_mle_8q9q10q11q_failure_vs_samples.png"
    svg_path = output_dir / "period_mle_8q9q10q11q_failure_vs_samples.svg"
    pdf_path = output_dir / "period_mle_8q9q10q11q_failure_vs_samples.pdf"
    # DeepSeek-style blue palette: vary only luminance/saturation by n.
    colors = ("#173B8F", "#2E5AAC", "#4D6BFE", "#86A5FF")
    plotted_points: list[tuple[SuccessPoint, ...]] = []
    for item in series:
        prefix: list[SuccessPoint] = []
        for point in item.points:
            if point.failure_rate <= 0.0:
                break
            prefix.append(point)
        plotted_points.append(tuple(prefix))
    positive_rates = [
        point.failure_rate
        for points in plotted_points
        for point in points
    ]
    if not positive_rates:
        raise ValueError("cannot draw a logarithmic axis when every failure rate is zero")
    lower_limit = 10.0 ** math.floor(math.log10(min(positive_rates)))
    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for color, item, points in zip(colors, series, plotted_points, strict=False):
        # The first zero observed failure is the end of the empirical curve:
        # subsequent values are not rendered because log(0) is undefined.
        ax.plot(
            [point.sample_count for point in points],
            [point.failure_rate for point in points],
            color=color,
            marker="o",
            linewidth=2.2,
            markersize=5.2,
            label=f"n={item.nqubit}",
        )

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlim(float(PLOT_X_MIN), float(PLOT_X_MAX))
    ax.set_xticks(
        sorted({PLOT_X_MIN, *range(250, PLOT_X_MAX, 250), PLOT_X_MAX})
    )
    ax.set_xlabel("samples")
    ax.set_ylabel("Fail rate")
    ax.set_title("Maximum Likelihood Estimation")
    ax.set_ylim(lower_limit, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, svg_path, pdf_path


def run_nqubit_experiment(
    *,
    nqubit: int,
    model_dir: Path,
    output_dir: Path,
    sample_counts: tuple[int, ...],
    trials_per_period: int,
    trial_batch_size: int,
    period_min: int | None,
    period_max: int | None,
    shift: int,
    exact_support_override: bool | None,
    seed: int,
) -> SuccessSeries:
    checkpoint_path, phase_path = resolve_artifact_paths(model_dir, nqubit)
    phase_payload = load_json_object(phase_path)
    stored_nqubit = phase_payload.get("nqubit")
    if stored_nqubit != nqubit:
        raise ValueError(
            f"nqubit mismatch: requested {nqubit}, metadata contains {stored_nqubit}"
        )

    checkpoint_phases = load_checkpoint_phases(checkpoint_path)
    stored_phases = metadata_phases(phase_payload, phase_path)
    if len(checkpoint_phases) != len(stored_phases) or not np.allclose(
        checkpoint_phases,
        stored_phases,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(f"checkpoint phases do not match metadata for n={nqubit}")

    periods = resolve_periods(
        phase_payload,
        period_min=period_min,
        period_max=period_max,
    )
    if shift < 0 or shift >= min(periods):
        raise ValueError(
            f"shift must satisfy 0 <= shift < min(periods)={min(periods)}"
        )

    stored_exact_support = phase_payload.get("exact_support", False)
    if not isinstance(stored_exact_support, bool):
        raise TypeError(f"invalid exact_support in {phase_path}")
    exact_support = (
        stored_exact_support
        if exact_support_override is None
        else exact_support_override
    )

    print(
        "config "
        f"nqubit={nqubit} periods={periods[0]}..{periods[-1]} "
        f"candidates={len(periods)} shift={shift} "
        f"exact_support={exact_support} "
        f"sample_counts={list(sample_counts)} "
        f"trials_per_period={trials_per_period}",
        flush=True,
    )
    probabilities = exact_probability_table(
        nqubit=nqubit,
        phases=checkpoint_phases,
        periods=periods,
        shift=shift,
        exact_support=exact_support,
    )
    probability_path = save_probability_table(
        output_dir,
        nqubit=nqubit,
        periods=periods,
        probabilities=probabilities,
        shift=shift,
        exact_support=exact_support,
    )
    points = evaluate_mle_success(
        probabilities,
        sample_counts=sample_counts,
        trials_per_period=trials_per_period,
        trial_batch_size=trial_batch_size,
        seed=seed,
    )
    return SuccessSeries(
        nqubit=nqubit,
        periods=periods,
        period_min=periods[0],
        period_max=periods[-1],
        candidate_count=len(periods),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        phase_path=str(phase_path),
        phase_sha256=sha256_file(phase_path),
        probability_path=str(probability_path),
        shift=shift,
        exact_support=exact_support,
        seed=seed,
        points=tuple(points),
    )


def main() -> None:
    args = parse_args()
    if args.trials_per_period < 1:
        raise ValueError("trials-per-period must be positive")
    if args.trial_batch_size < 1:
        raise ValueError("trial-batch-size must be positive")

    nqubits = tuple(dict.fromkeys(int(value) for value in args.nqubits))
    if not nqubits or any(value < 2 for value in nqubits):
        raise ValueError("nqubits must contain values of at least 2")
    sample_counts = resolve_sample_counts(args.sample_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "nqubits": list(nqubits),
        "shift": args.shift,
        "exact_support_override": args.exact_support,
        "probability_source": (
            "exact Qiskit Statevector evolution; p(x|r)=|psi_r(x)|^2; "
            "no shot-based probability estimation"
        ),
        "estimator": "argmax_r sum_i log p(x_i|r)",
        "tie_break": "lowest candidate-period index",
        "sample_counts": list(sample_counts),
        "trials_per_period": args.trials_per_period,
        "nested_measurement_samples": True,
        "base_seed": args.seed,
        "reported_metric": "failure_rate",
        "failure_rate_definition": "1 - success_rate",
        "x_axis_scale": "linear",
        "x_axis_limits": [PLOT_X_MIN, PLOT_X_MAX],
        "y_axis_scale": "log",
    }
    series = [
        run_nqubit_experiment(
            nqubit=nqubit,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            sample_counts=sample_counts,
            trials_per_period=args.trials_per_period,
            trial_batch_size=args.trial_batch_size,
            period_min=args.period_min,
            period_max=args.period_max,
            shift=args.shift,
            exact_support_override=args.exact_support,
            seed=args.seed + nqubit * 1_000_003,
        )
        for nqubit in nqubits
    ]
    json_path, csv_path = save_results(
        args.output_dir,
        metadata=metadata,
        series=series,
    )
    png_path, svg_path, pdf_path = save_plot(
        args.output_dir,
        series=series,
    )

    for item in series:
        for point in item.points:
            print(
                f"n={item.nqubit} samples={point.sample_count:4d} "
                f"failure_rate={point.failure_rate:.6f} "
                f"failed={point.failed}/{point.total}",
            )
        print(f"n={item.nqubit} probability_path={item.probability_path}")
    print(f"json_path={json_path}")
    print(f"csv_path={csv_path}")
    print(f"png_path={png_path}")
    print(f"svg_path={svg_path}")
    print(f"pdf_path={pdf_path}")


if __name__ == "__main__":
    main()
