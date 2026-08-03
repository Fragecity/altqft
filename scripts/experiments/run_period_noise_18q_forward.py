from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from altqft.circuits.HPgenerators import HP1_shared_parameter
from altqft.nn.period_decoder import (
    DECODER_TYPE,
    DeepSetPeriodPredictor,
    predictor_from_checkpoint,
)
from altqft.nn.process_qc import _torch_circuit_probability_vectors

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PHASE_PATH = Path("model/ph1_hp1_shared_fi_shift_18q_p2-511_phases.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "model/period_recovery_distribution_18q_p2-511_nibble_ddp/selected.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/noise_18q_nibble")


@dataclass(frozen=True, slots=True)
class NoisePoint:
    noise_strength: float
    accuracy: float
    correct: int
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only 18q period-recovery evaluation under the same post-circuit "
            "global depolarizing noise model used in the paper."
        )
    )
    parser.add_argument("--phase-path", type=Path, default=DEFAULT_PHASE_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise-min", type=float, default=1e-3)
    parser.add_argument("--noise-max", type=float, default=1e-2)
    parser.add_argument("--noise-count", type=int, default=10)
    parser.add_argument(
        "--shots",
        type=int,
        default=1024 * 18 * 18,
        help="Measurement samples per period/noise draw. Use 0 for exact weighted distributions.",
    )
    parser.add_argument("--draws-per-period", type=int, default=1)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument(
        "--exact-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact periodic support, matching the paper's noise evaluation.",
    )
    parser.add_argument("--period-chunk-size", type=int, default=32)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for circuit simulation and NN forward.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _load_phase_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid phase payload: {path}")
    return payload


def _load_model(
    checkpoint_path: Path,
    *,
    nqubit: int,
    device: torch.device,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...], dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    model, candidate_periods = predictor_from_checkpoint(payload, nqubit=nqubit)
    if model.architecture != "weighted":
        raise ValueError("18q checkpoint is expected to use weighted architecture")
    model.eval()
    model.to(device)
    return model, tuple(int(value) for value in candidate_periods), payload


def _basis_bits_chunk(start: int, stop: int, nqubit: int, device: torch.device) -> torch.Tensor:
    columns = torch.arange(start, stop, device=device, dtype=torch.long)
    bit_positions = torch.arange(nqubit - 1, -1, -1, device=device, dtype=torch.long)
    return ((columns[:, None] >> bit_positions) & 1).to(torch.long)


def _precompute_feature_bank(
    model: DeepSetPeriodPredictor,
    *,
    nqubit: int,
    feature_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    size = 1 << nqubit
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, size, feature_chunk_size):
            stop = min(size, start + feature_chunk_size)
            bits = _basis_bits_chunk(start, stop, nqubit, device).unsqueeze(0)
            features = model._weighted_features(bits).squeeze(0)
            chunks.append(features.detach())
    return torch.cat(chunks, dim=0).contiguous()


def _predict_periods_from_pooled_features(
    model: DeepSetPeriodPredictor,
    pooled: torch.Tensor,
) -> torch.Tensor:
    hidden = model.head_norm(pooled)
    for block in model.head_blocks:
        hidden = block(hidden)
    predicted, _, _ = model.decode_topk_from_pooled_features(hidden, 1)
    return predicted[:, 0]


def _noise_levels(noise_min: float, noise_max: float, count: int) -> tuple[float, ...]:
    if noise_min <= 0.0 or noise_max <= 0.0:
        raise ValueError("noise bounds must be positive")
    if count < 1:
        raise ValueError("noise-count must be positive")
    return tuple(float(value) for value in np.geomspace(noise_min, noise_max, count))


def _sample_weight_rows(
    probabilities: np.ndarray,
    *,
    shots: int,
    draws_per_period: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = np.empty(
        (probabilities.shape[0] * draws_per_period, probabilities.shape[1]),
        dtype=np.float32,
    )
    row_index = 0
    for distribution in probabilities:
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = np.clip(distribution / distribution.sum(), 0.0, None)
        distribution /= distribution.sum()
        for _ in range(draws_per_period):
            if shots > 0:
                counts = rng.multinomial(shots, distribution)
                rows[row_index] = counts.astype(np.float32) / float(shots)
            else:
                rows[row_index] = distribution.astype(np.float32)
            row_index += 1
    return rows


def _write_outputs(
    output_dir: Path,
    *,
    metadata: dict[str, object],
    points: list[NoisePoint],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "period_noise_18q_forward_results.json"
    csv_path = output_dir / "period_noise_18q_forward_results.csv"
    png_path = output_dir / "period_noise_18q_accuracy_vs_noise.png"
    svg_path = output_dir / "period_noise_18q_accuracy_vs_noise.svg"

    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "points": [asdict(point) for point in points],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("noise_strength,accuracy,correct,total\n")
        for point in points:
            handle.write(
                f"{point.noise_strength:.12g},{point.accuracy:.12g},"
                f"{point.correct},{point.total}\n"
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        [point.noise_strength for point in points],
        [point.accuracy for point in points],
        marker="o",
        linewidth=2.0,
        markersize=5.0,
        color="#1f7668",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Global depolarizing noise strength")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return json_path, csv_path, png_path, svg_path


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    phase_payload = _load_phase_payload(args.phase_path)
    nqubit_value = phase_payload.get("nqubit")
    if not isinstance(nqubit_value, int):
        raise ValueError(f"invalid nqubit in {args.phase_path}")
    nqubit = nqubit_value
    phases = phase_payload["phases"]
    if not isinstance(phases, list) or not all(isinstance(value, (int, float)) for value in phases):
        raise ValueError(f"invalid phases in {args.phase_path}")
    circuit = HP1_shared_parameter(nqubit, [float(value) for value in phases])
    model, candidate_periods, checkpoint_payload = _load_model(
        args.checkpoint_path,
        nqubit=nqubit,
        device=device,
    )
    if tuple(candidate_periods) != tuple(range(candidate_periods[0], candidate_periods[-1] + 1)):
        raise ValueError("this script expects a contiguous candidate-period range")

    noise_levels = _noise_levels(args.noise_min, args.noise_max, args.noise_count)
    size = 1 << nqubit
    rng = np.random.default_rng(args.seed)

    print(
        "config "
        f"device={device} nqubit={nqubit} periods={candidate_periods[0]}..{candidate_periods[-1]} "
        f"shots={args.shots} draws_per_period={args.draws_per_period} "
        f"exact_support={args.exact_support} shift={args.shift}"
    )
    print("precomputing weighted DeepSet features")
    feature_bank = _precompute_feature_bank(
        model,
        nqubit=nqubit,
        feature_chunk_size=args.feature_chunk_size,
        device=device,
    )
    print(f"feature_bank shape={tuple(feature_bank.shape)}")

    correct = np.zeros(len(noise_levels), dtype=np.int64)
    total = len(candidate_periods) * args.draws_per_period
    uniform_weight = 1.0 / float(size)

    with torch.inference_mode():
        for start in range(0, len(candidate_periods), args.period_chunk_size):
            period_chunk = candidate_periods[start : start + args.period_chunk_size]
            stop = start + len(period_chunk)
            probabilities = _torch_circuit_probability_vectors(
                circuit,
                period_chunk,
                args.shift,
                exact_support=args.exact_support,
                device=device,
            )
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            probabilities_cpu = probabilities.detach().cpu().to(torch.float64).numpy()
            labels = torch.tensor(period_chunk, device=device, dtype=torch.long)
            labels = labels.repeat_interleave(args.draws_per_period)

            for noise_index, noise_strength in enumerate(noise_levels):
                noisy = (1.0 - noise_strength) * probabilities_cpu + noise_strength * uniform_weight
                noisy = np.clip(noisy, 0.0, None)
                noisy /= noisy.sum(axis=1, keepdims=True)
                weights_cpu = _sample_weight_rows(
                    noisy,
                    shots=args.shots,
                    draws_per_period=args.draws_per_period,
                    rng=rng,
                )
                weights = torch.from_numpy(weights_cpu).to(device=device)
                pooled = weights @ feature_bank
                predicted = _predict_periods_from_pooled_features(model, pooled)
                correct[noise_index] += int((predicted == labels).sum().item())

            print(
                f"processed periods {period_chunk[0]}..{period_chunk[-1]} "
                f"({stop}/{len(candidate_periods)})"
            )

    points = [
        NoisePoint(
            noise_strength=float(noise_strength),
            accuracy=float(correct[index] / float(total)),
            correct=int(correct[index]),
            total=int(total),
        )
        for index, noise_strength in enumerate(noise_levels)
    ]
    metadata = {
        "phase_path": str(args.phase_path),
        "checkpoint_path": str(args.checkpoint_path),
        "nqubit": nqubit,
        "period_min": candidate_periods[0],
        "period_max": candidate_periods[-1],
        "num_periods": len(candidate_periods),
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
        "noise_count": args.noise_count,
        "shots": args.shots,
        "draws_per_period": args.draws_per_period,
        "shift": args.shift,
        "exact_support": args.exact_support,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_metrics": checkpoint_payload.get("metrics"),
        "decoder_type": DECODER_TYPE,
    }
    json_path, csv_path, png_path, svg_path = _write_outputs(
        args.output_dir,
        metadata=metadata,
        points=points,
    )

    print(f"json_path={json_path}")
    print(f"csv_path={csv_path}")
    print(f"png_path={png_path}")
    print(f"svg_path={svg_path}")
    for point in points:
        print(
            f"noise={point.noise_strength:.6g} "
            f"accuracy={point.accuracy:.6f} correct={point.correct}/{point.total}"
        )


if __name__ == "__main__":
    main()
