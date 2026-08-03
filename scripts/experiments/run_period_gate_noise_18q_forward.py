from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from qiskit import QuantumCircuit

from altqft.circuits.HPgenerators import HP1_shared_parameter
from altqft.nn.period_decoder import (
    DECODER_TYPE,
    DeepSetPeriodPredictor,
    predictor_from_checkpoint,
)
from altqft.nn.process_qc import _torch_exact_support_indices, _torch_surrogate_support_indices
from altqft.nn.unitary_rows import (
    apply_controlled_phase_state_batch,
    apply_hadamard_state_batch,
    paired_row_indices,
    phase_row_indices,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PHASE_PATH = Path("model/ph1_hp1_shared_fi_shift_18q_p2-511_phases.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "model/period_recovery_distribution_18q_p2-511_nibble_ddp/selected.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/noise_18q_gate_nibble")


@dataclass(frozen=True, slots=True)
class GateNoisePoint:
    noise_strength: float
    accuracy: float
    correct: int
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only 18q period recovery under gate-level stochastic Pauli "
            "depolarizing noise inserted after each Qiskit HP1_shared gate."
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
        help="Measurement samples per period/noise draw. Use 0 for exact averaged noisy distributions.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=4,
        help="Monte Carlo noisy-circuit trajectories per period and noise point.",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=8192,
        help="Retain the largest sampled count patterns, matching nibble-decoder training.",
    )
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument(
        "--exact-support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--period-chunk-size", type=int, default=8)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _noise_levels(noise_min: float, noise_max: float, count: int) -> tuple[float, ...]:
    if noise_min <= 0.0 or noise_max <= 0.0:
        raise ValueError("noise bounds must be positive")
    if count < 1:
        raise ValueError("noise-count must be positive")
    return tuple(float(value) for value in np.geomspace(noise_min, noise_max, count))


def _load_phase_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid phase payload: {path}")
    return payload


def _load_model(
    checkpoint_path: Path,
    *,
    nqubit: int,
    device: torch.device,
) -> tuple[DeepSetPeriodPredictor, tuple[int, ...], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    model, candidate_periods = predictor_from_checkpoint(payload, nqubit=nqubit)
    if model.architecture != "weighted":
        raise ValueError("18q checkpoint is expected to use weighted architecture")
    model.eval()
    model.to(device)
    return model, candidate_periods, payload


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
            chunks.append(model._weighted_features(bits).squeeze(0).detach())
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


def _periodic_state_batch(
    *,
    nqubit: int,
    periods: tuple[int, ...],
    shift: int,
    trajectories: int,
    exact_support: bool,
    device: torch.device,
) -> torch.Tensor:
    size = 1 << nqubit
    states = torch.zeros(
        (len(periods) * trajectories, size),
        dtype=torch.complex64,
        device=device,
    )
    for period_index, period in enumerate(periods):
        support = (
            _torch_exact_support_indices(size, period, shift, device=device)
            if exact_support
            else _torch_surrogate_support_indices(size, period, shift, device=device)
        )
        amplitude = torch.tensor(
            1.0 / math.sqrt(float(support.numel())),
            dtype=torch.complex64,
            device=device,
        )
        for trajectory in range(trajectories):
            row = period_index * trajectories + trajectory
            states[row].index_fill_(0, support, amplitude)
    return states


def _selected_rows(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.flatnonzero(mask).astype(np.int64)).to(device=device)


def _apply_x_to_selected(states: torch.Tensor, rows: torch.Tensor, qubit: int) -> None:
    if rows.numel() == 0:
        return
    indices = torch.arange(states.shape[1], device=states.device, dtype=torch.long)
    flipped = indices ^ (1 << qubit)
    states[rows] = states.index_select(0, rows).index_select(1, flipped)


def _apply_z_to_selected(states: torch.Tensor, rows: torch.Tensor, qubit: int) -> None:
    if rows.numel() == 0:
        return
    active = phase_row_indices(states.shape[1], qubit, qubit, states.device)
    states[rows[:, None], active[None, :]] *= -1


def _apply_y_to_selected(states: torch.Tensor, rows: torch.Tensor, qubit: int) -> None:
    if rows.numel() == 0:
        return
    lower_rows, upper_rows = paired_row_indices(states.shape[1], qubit, states.device)
    selected = states.index_select(0, rows)
    lower_values = selected.index_select(1, lower_rows)
    upper_values = selected.index_select(1, upper_rows)
    selected = selected.clone()
    selected.index_copy_(1, lower_rows, -1j * upper_values)
    selected.index_copy_(1, upper_rows, 1j * lower_values)
    states[rows] = selected


def _apply_pauli_to_selected(
    states: torch.Tensor,
    rows: torch.Tensor,
    qubit: int,
    pauli: str,
) -> None:
    if pauli == "I" or rows.numel() == 0:
        return
    if pauli == "X":
        _apply_x_to_selected(states, rows, qubit)
        return
    if pauli == "Y":
        _apply_y_to_selected(states, rows, qubit)
        return
    if pauli == "Z":
        _apply_z_to_selected(states, rows, qubit)
        return
    raise ValueError(f"unsupported Pauli: {pauli}")


def _apply_one_qubit_depolarizing(
    states: torch.Tensor,
    *,
    qubit: int,
    noise_strength: float,
    rng: np.random.Generator,
) -> None:
    draws = rng.random(states.shape[0])
    error_mask = draws < noise_strength
    if not np.any(error_mask):
        return
    paulis = rng.integers(0, 3, size=int(error_mask.sum()))
    error_rows = np.flatnonzero(error_mask)
    for pauli_index, pauli in enumerate(("X", "Y", "Z")):
        selected = error_rows[paulis == pauli_index]
        _apply_pauli_to_selected(
            states,
            torch.from_numpy(selected.astype(np.int64)).to(device=states.device),
            qubit,
            pauli,
        )


def _apply_two_qubit_depolarizing(
    states: torch.Tensor,
    *,
    control: int,
    target: int,
    noise_strength: float,
    rng: np.random.Generator,
) -> None:
    draws = rng.random(states.shape[0])
    error_mask = draws < noise_strength
    if not np.any(error_mask):
        return
    pauli_pairs = (
        ("I", "X"),
        ("I", "Y"),
        ("I", "Z"),
        ("X", "I"),
        ("X", "X"),
        ("X", "Y"),
        ("X", "Z"),
        ("Y", "I"),
        ("Y", "X"),
        ("Y", "Y"),
        ("Y", "Z"),
        ("Z", "I"),
        ("Z", "X"),
        ("Z", "Y"),
        ("Z", "Z"),
    )
    pair_indices = rng.integers(0, len(pauli_pairs), size=int(error_mask.sum()))
    error_rows = np.flatnonzero(error_mask)
    for pair_index, (control_pauli, target_pauli) in enumerate(pauli_pairs):
        selected = error_rows[pair_indices == pair_index]
        rows = torch.from_numpy(selected.astype(np.int64)).to(device=states.device)
        _apply_pauli_to_selected(states, rows, control, control_pauli)
        _apply_pauli_to_selected(states, rows, target, target_pauli)


def _qubit_index(circuit: QuantumCircuit, qubit: Any) -> int:
    return int(circuit.find_bit(qubit).index)


def _apply_noisy_qiskit_circuit_batch(
    circuit: QuantumCircuit,
    states: torch.Tensor,
    *,
    noise_strength: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    for instruction in circuit.data:
        operation = instruction.operation
        op_name = operation.name.lower()
        if op_name == "h":
            qubit = _qubit_index(circuit, instruction.qubits[0])
            states = apply_hadamard_state_batch(states, qubit)
            _apply_one_qubit_depolarizing(
                states,
                qubit=qubit,
                noise_strength=noise_strength,
                rng=rng,
            )
            continue
        if op_name == "cp":
            control = _qubit_index(circuit, instruction.qubits[0])
            target = _qubit_index(circuit, instruction.qubits[1])
            theta = float(operation.params[0])
            states = apply_controlled_phase_state_batch(states, control, target, theta)
            _apply_two_qubit_depolarizing(
                states,
                control=control,
                target=target,
                noise_strength=noise_strength,
                rng=rng,
            )
            continue
        raise ValueError(f"unsupported Qiskit operation in HP circuit: {operation.name}")
    return states


def _retain_largest_patterns(weights: np.ndarray, max_patterns: int) -> np.ndarray:
    if max_patterns < 1:
        raise ValueError("max_patterns must be positive")
    retained = np.zeros_like(weights, dtype=np.float64)
    nonzero = np.flatnonzero(weights)
    if nonzero.size <= max_patterns:
        retained[nonzero] = weights[nonzero]
    else:
        selected_offsets = np.argpartition(weights[nonzero], -max_patterns)[
            -max_patterns:
        ]
        selected = nonzero[selected_offsets]
        retained[selected] = weights[selected]
    retained /= retained.sum()
    return retained


def _sample_weight_rows(
    probabilities: np.ndarray,
    *,
    shots: int,
    max_patterns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = np.empty_like(probabilities, dtype=np.float32)
    for row, distribution in enumerate(probabilities):
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = np.clip(distribution / distribution.sum(), 0.0, None)
        distribution /= distribution.sum()
        weights = (
            rng.multinomial(shots, distribution).astype(np.float64)
            if shots > 0
            else distribution
        )
        rows[row] = _retain_largest_patterns(weights, max_patterns).astype(np.float32)
    return rows


def _write_outputs(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    points: list[GateNoisePoint],
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "period_gate_noise_18q_forward_results.json"
    csv_path = output_dir / "period_gate_noise_18q_forward_results.csv"
    png_path = output_dir / "period_gate_noise_18q_accuracy_vs_noise.png"
    svg_path = output_dir / "period_gate_noise_18q_accuracy_vs_noise.svg"
    pdf_path = output_dir / "period_gate_noise_18q_accuracy_vs_noise.pdf"

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
        color="#b45309",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Noise strength")
    ax.set_xticks([10**-3, 10**-2.5, 10**-2, 10**-1.5])
    ax.set_xticklabels(
        [r"$10^{-3}$", r"$10^{-2.5}$", r"$10^{-2}$", r"$10^{-1.5}$"]
    )
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return json_path, csv_path, png_path, svg_path, pdf_path


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    phase_payload = _load_phase_payload(args.phase_path)
    nqubit = int(phase_payload["nqubit"])
    phases = phase_payload["phases"]
    if not isinstance(phases, list) or not all(isinstance(value, (int, float)) for value in phases):
        raise ValueError(f"invalid phases in {args.phase_path}")
    circuit = HP1_shared_parameter(nqubit, [float(value) for value in phases])
    model, candidate_periods, checkpoint_payload = _load_model(
        args.checkpoint_path,
        nqubit=nqubit,
        device=device,
    )
    noise_levels = _noise_levels(args.noise_min, args.noise_max, args.noise_count)
    if args.trajectories < 1:
        raise ValueError("trajectories must be positive")
    if args.max_patterns < 1:
        raise ValueError("max-patterns must be positive")

    print(
        "config "
        f"device={device} nqubit={nqubit} periods={candidate_periods[0]}..{candidate_periods[-1]} "
        f"noise={noise_levels[0]:.6g}..{noise_levels[-1]:.6g} "
        f"trajectories={args.trajectories} shots={args.shots} "
        f"gates={len(circuit.data)} exact_support={args.exact_support}"
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
    total = len(candidate_periods)
    with torch.inference_mode():
        for noise_index, noise_strength in enumerate(noise_levels):
            noise_rng = np.random.default_rng(args.seed + (noise_index + 1) * 1_000_003)
            shot_rng = np.random.default_rng(args.seed + (noise_index + 1) * 2_000_003)
            for start in range(0, len(candidate_periods), args.period_chunk_size):
                period_chunk = candidate_periods[start : start + args.period_chunk_size]
                states = _periodic_state_batch(
                    nqubit=nqubit,
                    periods=period_chunk,
                    shift=args.shift,
                    trajectories=args.trajectories,
                    exact_support=args.exact_support,
                    device=device,
                )
                states = _apply_noisy_qiskit_circuit_batch(
                    circuit,
                    states,
                    noise_strength=float(noise_strength),
                    rng=noise_rng,
                )
                probabilities = states.abs().pow(2).reshape(
                    len(period_chunk),
                    args.trajectories,
                    -1,
                )
                probabilities = probabilities.mean(dim=1)
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
                weights_cpu = _sample_weight_rows(
                    probabilities.detach().cpu().to(torch.float64).numpy(),
                    shots=args.shots,
                    max_patterns=args.max_patterns,
                    rng=shot_rng,
                )
                weights = torch.from_numpy(weights_cpu).to(device=device)
                pooled = weights @ feature_bank
                predicted = _predict_periods_from_pooled_features(model, pooled)
                labels = torch.tensor(period_chunk, device=device, dtype=torch.long)
                correct[noise_index] += int((predicted == labels).sum().item())

            print(
                f"noise={noise_strength:.6g} "
                f"accuracy={correct[noise_index] / float(total):.6f} "
                f"correct={correct[noise_index]}/{total}"
            )

    points = [
        GateNoisePoint(
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
        "trajectories": args.trajectories,
        "max_patterns": args.max_patterns,
        "shift": args.shift,
        "exact_support": args.exact_support,
        "seed": args.seed,
        "device": str(device),
        "qiskit_gate_count": len(circuit.data),
        "noise_model": (
            "stochastic Pauli depolarizing after each Qiskit h/cp gate; "
            "1q errors sample X/Y/Z, 2q errors sample non-identity two-qubit Paulis"
        ),
        "decoder_type": DECODER_TYPE,
        "beam_width": model.beam_width,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_metrics": checkpoint_payload.get("metrics"),
    }
    json_path, csv_path, png_path, svg_path, pdf_path = _write_outputs(
        args.output_dir,
        metadata=metadata,
        points=points,
    )
    print(f"json_path={json_path}")
    print(f"csv_path={csv_path}")
    print(f"png_path={png_path}")
    print(f"svg_path={svg_path}")
    print(f"pdf_path={pdf_path}")


if __name__ == "__main__":
    main()
