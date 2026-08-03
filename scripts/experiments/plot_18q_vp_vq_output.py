#!/usr/bin/env python3
"""Sample and plot two 18-qubit HP1_shared output distributions.

The input ``|V_r>`` is the normalized uniform superposition over
``0, r, 2r, ... < 2**n``.  The two requested periods are evolved and sampled
in independent worker processes so that separate CUDA devices can run them
concurrently.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import torch

from altqft.circuits.HPgenerators import HP1_shared_parameter
from altqft.nn.process_qc import _torch_circuit_probability_vector


DEFAULT_PHASE_PATH = Path(
    "model/ph1_hp1_shared_fi_shift_18q_p2-511_phases.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/18q_vp_vq_p3_q5")
DEFAULT_SHOTS = 100_000_000
DEFAULT_SEED = 20260724


@dataclass(frozen=True)
class SimulationTask:
    period: int
    role: str
    device: str
    seed: int
    shots: int
    phase_path: str


@dataclass(frozen=True)
class SimulationResult:
    period: int
    role: str
    device: str
    seed: int
    shots: int
    support_count: int
    probabilities_exact: np.ndarray
    counts: np.ndarray
    started_unix: float
    finished_unix: float
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained 18q HP1_shared circuit on |V_p> and |V_q>, "
            "sample both output distributions in parallel, and plot P(x)."
        )
    )
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phase-path", type=Path, default=DEFAULT_PHASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated devices for p and q, for example cuda:0,cuda:1. "
            "By default the first two CUDA devices are used, or two CPU workers."
        ),
    )
    return parser.parse_args()


def _default_devices() -> tuple[str, str]:
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "cuda:0", "cuda:1"
    if torch.cuda.is_available():
        return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def _parse_devices(raw_devices: str | None) -> tuple[str, str]:
    if raw_devices is None:
        return _default_devices()
    devices = tuple(value.strip() for value in raw_devices.split(","))
    if len(devices) != 2 or not all(devices):
        raise ValueError("--devices must contain exactly two comma-separated devices")
    return devices[0], devices[1]


def _load_phase_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid parameter payload in {path}")
    if payload.get("nqubit") != 18:
        raise ValueError(f"{path} is not an 18-qubit parameter file")
    phases = payload.get("phases")
    if not isinstance(phases, list) or not all(
        isinstance(value, (int, float)) for value in phases
    ):
        raise ValueError(f"invalid phases in {path}")
    return cast(dict[str, Any], payload)


def _validate_period(period: int, *, name: str, size: int) -> None:
    if period <= 0 or period >= size:
        raise ValueError(f"{name} must satisfy 0 < {name} < {size}")
    if period % 2 == 0:
        raise ValueError(f"{name} must be odd")


def _simulate_and_sample(task: SimulationTask) -> SimulationResult:
    started_unix = time.time()
    phase_path = Path(task.phase_path)
    phase_payload = _load_phase_payload(phase_path)
    nqubit = int(phase_payload["nqubit"])
    size = 1 << nqubit
    support_count = ((size - 1) // task.period) + 1
    phases = [float(value) for value in phase_payload["phases"]]
    circuit = HP1_shared_parameter(nqubit, phases)
    device = torch.device(task.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    with torch.inference_mode():
        probabilities = _torch_circuit_probability_vector(
            circuit,
            task.period,
            0,
            exact_support=True,
            device=device,
        )
        probabilities = probabilities / probabilities.sum()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        probabilities_exact = (
            probabilities.detach().cpu().to(torch.float64).numpy()
        )

    probabilities_exact = np.maximum(probabilities_exact, 0.0)
    probabilities_exact /= probabilities_exact.sum(dtype=np.float64)
    rng = np.random.default_rng(task.seed)
    counts = rng.multinomial(task.shots, probabilities_exact)
    finished_unix = time.time()

    return SimulationResult(
        period=task.period,
        role=task.role,
        device=task.device,
        seed=task.seed,
        shots=task.shots,
        support_count=support_count,
        probabilities_exact=probabilities_exact,
        counts=counts,
        started_unix=started_unix,
        finished_unix=finished_unix,
        elapsed_seconds=finished_unix - started_unix,
    )


def _sampling_metrics(result: SimulationResult) -> dict[str, Any]:
    empirical = result.counts.astype(np.float64) / float(result.shots)
    difference = empirical - result.probabilities_exact
    exact_max_x = int(np.argmax(result.probabilities_exact))
    sampled_max_x = int(np.argmax(empirical))
    return {
        "period": result.period,
        "device": result.device,
        "seed": result.seed,
        "shots": result.shots,
        "input_support_count": result.support_count,
        "count_sum": int(result.counts.sum()),
        "exact_probability_sum": float(result.probabilities_exact.sum()),
        "sampled_probability_sum": float(empirical.sum()),
        "nonzero_sample_bins": int(np.count_nonzero(result.counts)),
        "sampling_total_variation": float(0.5 * np.abs(difference).sum()),
        "sampling_max_absolute_error": float(np.abs(difference).max()),
        "exact_peak_x": exact_max_x,
        "exact_peak_probability": float(result.probabilities_exact[exact_max_x]),
        "sampled_peak_x": sampled_max_x,
        "sampled_peak_probability": float(empirical[sampled_max_x]),
        "started_utc": datetime.fromtimestamp(
            result.started_unix, tz=timezone.utc
        ).isoformat(),
        "finished_utc": datetime.fromtimestamp(
            result.finished_unix, tz=timezone.utc
        ).isoformat(),
        "elapsed_seconds": result.elapsed_seconds,
    }


def _curve_comparison(
    p_result: SimulationResult,
    q_result: SimulationResult,
) -> dict[str, float]:
    p_exact = p_result.probabilities_exact
    q_exact = q_result.probabilities_exact
    p_empirical = p_result.counts.astype(np.float64) / float(p_result.shots)
    q_empirical = q_result.counts.astype(np.float64) / float(q_result.shots)
    exact_coefficient = float(np.sqrt(p_exact * q_exact).sum())
    empirical_coefficient = float(np.sqrt(p_empirical * q_empirical).sum())
    return {
        "exact_total_variation": float(0.5 * np.abs(p_exact - q_exact).sum()),
        "sampled_total_variation": float(
            0.5 * np.abs(p_empirical - q_empirical).sum()
        ),
        "exact_bhattacharyya_coefficient": exact_coefficient,
        "sampled_bhattacharyya_coefficient": empirical_coefficient,
        "exact_hellinger_squared": float(max(0.0, 1.0 - exact_coefficient)),
        "sampled_hellinger_squared": float(
            max(0.0, 1.0 - empirical_coefficient)
        ),
    }


def _write_data(
    output_dir: Path,
    p_result: SimulationResult,
    q_result: SimulationResult,
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p_empirical = p_result.counts.astype(np.float64) / float(p_result.shots)
    q_empirical = q_result.counts.astype(np.float64) / float(q_result.shots)
    x = np.arange(p_result.counts.size, dtype=np.int64)

    npz_path = output_dir / "vp_vq_output_distributions.npz"
    np.savez_compressed(
        npz_path,
        x=x,
        p=np.asarray(p_result.period, dtype=np.int64),
        q=np.asarray(q_result.period, dtype=np.int64),
        shots=np.asarray(p_result.shots, dtype=np.int64),
        counts_p=p_result.counts,
        counts_q=q_result.counts,
        probability_p_sampled=p_empirical,
        probability_q_sampled=q_empirical,
        probability_p_exact=p_result.probabilities_exact,
        probability_q_exact=q_result.probabilities_exact,
    )

    csv_path = output_dir / "vp_vq_output_distributions.csv"
    table = np.column_stack(
        (
            x,
            p_result.counts,
            p_empirical,
            p_result.probabilities_exact,
            q_result.counts,
            q_empirical,
            q_result.probabilities_exact,
        )
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=(
            f"x,count_p{p_result.period},probability_p{p_result.period}_sampled,"
            f"probability_p{p_result.period}_exact,count_q{q_result.period},"
            f"probability_q{q_result.period}_sampled,"
            f"probability_q{q_result.period}_exact"
        ),
        comments="",
        fmt=("%d", "%d", "%.12e", "%.12e", "%d", "%.12e", "%.12e"),
    )

    metadata_path = output_dir / "vp_vq_output_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return npz_path, csv_path, metadata_path


def _plot_distributions(
    output_dir: Path,
    p_result: SimulationResult,
    q_result: SimulationResult,
) -> tuple[Path, Path]:
    x = np.arange(p_result.counts.size, dtype=np.int64)
    p_empirical = p_result.counts.astype(np.float64) / float(p_result.shots)
    q_empirical = q_result.counts.astype(np.float64) / float(q_result.shots)
    plotted_x = x[1:]

    fig, ax = plt.subplots(figsize=(14.0, 6.5), constrained_layout=True)
    ax.plot(
        plotted_x,
        p_empirical[1:],
        color="#0072B2",
        linewidth=0.55,
        alpha=0.82,
        rasterized=True,
        label=rf"$r = {p_result.period}$",
    )
    ax.plot(
        plotted_x,
        q_empirical[1:],
        color="#D55E00",
        linewidth=0.55,
        alpha=0.78,
        rasterized=True,
        label=rf"$r = {q_result.period}$",
    )
    ax.set_xlim(1, int(x[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xticks((1, 65_536, 131_072, 196_608, int(x[-1])))
    ax.set_xlabel("output", fontsize=20)
    ax.set_ylabel("Probability", fontsize=20)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    ax.grid(True, linewidth=0.45, alpha=0.3)
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=20)
    ax.margins(x=0.0)

    png_path = output_dir / "vp_vq_output_distributions.png"
    pdf_path = output_dir / "vp_vq_output_distributions.pdf"
    fig.savefig(png_path, dpi=260)
    fig.savefig(pdf_path, dpi=260)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    phase_path = args.phase_path.resolve()
    output_dir = args.output_dir.resolve()
    payload = _load_phase_payload(phase_path)
    nqubit = int(payload["nqubit"])
    size = 1 << nqubit
    _validate_period(args.p, name="p", size=size)
    _validate_period(args.q, name="q", size=size)
    if args.p == args.q:
        raise ValueError("p and q must be different")
    if args.shots < 1:
        raise ValueError("--shots must be positive")

    devices = _parse_devices(args.devices)
    tasks = (
        SimulationTask(
            period=args.p,
            role="p",
            device=devices[0],
            seed=args.seed,
            shots=args.shots,
            phase_path=str(phase_path),
        ),
        SimulationTask(
            period=args.q,
            role="q",
            device=devices[1],
            seed=args.seed + 1,
            shots=args.shots,
            phase_path=str(phase_path),
        ),
    )

    wall_started = time.time()
    results_by_role: dict[str, SimulationResult] = {}
    spawn_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn_context) as executor:
        future_to_task = {
            executor.submit(_simulate_and_sample, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results_by_role[task.role] = result
            print(
                f"completed role={task.role} period={task.period} "
                f"device={task.device} shots={task.shots:,} "
                f"elapsed={result.elapsed_seconds:.3f}s",
                flush=True,
            )
    wall_finished = time.time()

    p_result = results_by_role["p"]
    q_result = results_by_role["q"]
    overlap_seconds = max(
        0.0,
        min(p_result.finished_unix, q_result.finished_unix)
        - max(p_result.started_unix, q_result.started_unix),
    )
    metadata: dict[str, Any] = {
        "nqubit": nqubit,
        "size": size,
        "ansatz": payload.get("ansatz"),
        "objective": payload.get("objective"),
        "phase_path": str(phase_path),
        "phases": [float(value) for value in payload["phases"]],
        "input_state_definition": (
            "|V_r> = (1/sqrt(R_r)) sum_{k=0}^{R_r-1} |k r>, "
            "R_r = floor((2^n - 1)/r) + 1"
        ),
        "shift": 0,
        "exact_support": True,
        "sampling_method": "NumPy multinomial draw from the evolved distribution",
        "parallel_worker_count": 2,
        "parallel_wall_seconds": wall_finished - wall_started,
        "worker_overlap_seconds": overlap_seconds,
        "p": _sampling_metrics(p_result),
        "q": _sampling_metrics(q_result),
        "curve_comparison": _curve_comparison(p_result, q_result),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    npz_path, csv_path, metadata_path = _write_data(
        output_dir,
        p_result,
        q_result,
        metadata,
    )
    png_path, pdf_path = _plot_distributions(output_dir, p_result, q_result)

    print(f"npz={npz_path}")
    print(f"csv={csv_path}")
    print(f"metadata={metadata_path}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")


if __name__ == "__main__":
    main()
