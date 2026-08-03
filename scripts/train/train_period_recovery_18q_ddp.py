from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from qiskit import QuantumCircuit
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from altqft.circuits.HPgenerators import HP1_shared_parameter
from altqft.nn.period_decoder import (
    DECODER_TYPE,
    DEFAULT_BEAM_WIDTH,
    TOKEN_BITS,
    DeepSetPeriodPredictor,
    period_token_loss,
)
from altqft.nn.process_qc import (
    _apply_circuit_state_batch,
    _torch_exact_support_indices,
    _torch_surrogate_support_indices,
)

DEFAULT_PHASE_PATH = Path("model/ph1_hp1_shared_fi_shift_18q_p2-511_phases.json")
DEFAULT_RUN_DIR = Path("model/period_recovery_distribution_18q_p2-511_nibble_ddp")
DEFAULT_OUTPUT_DIR = Path("outputs/period_recovery_distribution_18q_p2-511_nibble_ddp")
DECODER_PARAMETER_PREFIXES = (
    "decoder_init",
    "token_embedding",
    "decoder_cell",
    "token_classifier",
)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    train_top4: float
    val_loss: float
    val_top1: float
    val_top4: float
    epoch_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the direct-binary 4-bit period decoder with one DDP process per GPU. "
            "Launch with torchrun --standalone --nproc_per_node=8."
        )
    )
    parser.add_argument("--phase-path", type=Path, default=DEFAULT_PHASE_PATH)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose matching encoder weights initialize the model.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--period-min", type=int, default=2)
    parser.add_argument("--period-max", type=int, default=511)
    parser.add_argument("--shots", type=int, default=1024 * 18 * 18)
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=8192,
        help="Largest number of nonzero shot-count patterns retained per example.",
    )
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--train-draws-per-epoch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1e-4,
        help="Lower fine-tuning rate for the transferred Deep Sets encoder.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=250)
    parser.add_argument("--validation-period-limit", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--latest-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exact-support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Freeze a checkpoint-initialized encoder for these initial epochs.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.period_min < 2 or args.period_max < args.period_min:
        raise ValueError("invalid period range")
    if args.period_max >= 1 << 18:
        raise ValueError("period_max must fit in 18 bits")
    for name in (
        "shots",
        "max_patterns",
        "local_batch_size",
        "train_draws_per_epoch",
        "epochs",
        "min_epochs",
        "early_stopping_patience",
        "log_interval",
        "latest_interval",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if args.freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs must be non-negative")
    if args.freeze_encoder_epochs > 0 and args.init_checkpoint is None:
        raise ValueError("freeze_encoder_epochs requires --init-checkpoint")
    if args.learning_rate <= 0.0 or args.encoder_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")


def init_distributed() -> DistributedContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required):
        raise RuntimeError(
            "distributed environment is missing; launch this script with torchrun"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 18q DDP training")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = True


def configure_logger(path: Path, *, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("altqft.train.period_recovery_18q_ddp")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_circuit(path: Path) -> tuple[QuantumCircuit, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid phase payload in {path}")
    nqubit = payload.get("nqubit")
    phases = payload.get("phases")
    if not isinstance(nqubit, int):
        raise ValueError(f"phase payload is missing nqubit in {path}")
    if not isinstance(phases, list) or not all(
        isinstance(value, (int, float)) for value in phases
    ):
        raise ValueError(f"phase payload is missing numeric phases in {path}")
    return HP1_shared_parameter(nqubit, [float(value) for value in phases]), nqubit


def initialize_encoder_from_checkpoint(
    model: DeepSetPeriodPredictor,
    checkpoint_path: Path | None,
) -> tuple[int, tuple[str, ...]]:
    if checkpoint_path is None:
        return 0, ()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"invalid initialization checkpoint: {checkpoint_path}")
    source = cast(dict[str, Tensor], payload["state_dict"])
    target = model.state_dict()
    transferable = {
        key: value
        for key, value in source.items()
        if not key.startswith(DECODER_PARAMETER_PREFIXES)
        and key in target
        and target[key].shape == value.shape
    }
    model.load_state_dict(transferable, strict=False)
    return len(transferable), tuple(sorted(transferable))


def clear_frozen_encoder_gradients(model: DeepSetPeriodPredictor) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith(DECODER_PARAMETER_PREFIXES):
            parameter.grad = None


def held_out_shift(period: int, seed: int) -> int:
    rng = np.random.default_rng(seed + period * 10_007)
    return int(rng.integers(0, period))


def training_periods(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    epoch: int,
    step: int,
) -> Tensor:
    period_count = args.period_max - args.period_min + 1
    global_batch_size = args.local_batch_size * context.world_size
    first_global_index = (
        (epoch - 1) * args.train_draws_per_epoch
        + step * global_batch_size
        + context.rank * args.local_batch_size
    )
    indices = torch.arange(args.local_batch_size, dtype=torch.long)
    offset = (epoch * 104_729) % period_count
    return cast(
        Tensor,
        args.period_min + (indices + first_global_index + offset) % period_count,
    )


def training_shifts(periods: Tensor, *, seed: int, epoch: int, step: int, rank: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch * 1_000_003 + step * 10_007 + rank * 101)
    shifts = torch.empty_like(periods)
    for index, period_value in enumerate(periods.tolist()):
        period = int(period_value)
        shift = int(torch.randint(period, (), generator=generator).item())
        reserved = held_out_shift(period, seed)
        shifts[index] = (shift + 1) % period if shift == reserved else shift
    return shifts


def periodic_state_batch(
    circuit: QuantumCircuit,
    periods: Tensor,
    shifts: Tensor,
    *,
    exact_support: bool,
    device: torch.device,
) -> Tensor:
    size = 1 << circuit.num_qubits
    states = torch.zeros(
        (periods.numel(), size),
        dtype=torch.complex64,
        device=device,
    )
    for row, (period, shift) in enumerate(zip(periods.tolist(), shifts.tolist())):
        support = (
            _torch_exact_support_indices(
                size,
                int(period),
                int(shift),
                device=device,
            )
            if exact_support
            else _torch_surrogate_support_indices(
                size,
                int(period),
                int(shift),
                device=device,
            )
        )
        amplitude = torch.tensor(
            1.0 / math.sqrt(float(support.numel())),
            dtype=torch.complex64,
            device=device,
        )
        states[row].index_fill_(0, support, amplitude)
    return states


def compress_measurements(
    columns: Tensor,
    *,
    nqubit: int,
    max_patterns: int,
) -> tuple[Tensor, Tensor]:
    batch_size = columns.shape[0]
    retained_columns = torch.zeros(
        (batch_size, max_patterns),
        dtype=torch.long,
        device=columns.device,
    )
    retained_weights = torch.zeros(
        (batch_size, max_patterns),
        dtype=torch.float32,
        device=columns.device,
    )
    for row in range(batch_size):
        unique_columns, counts = torch.unique(columns[row], return_counts=True)
        if unique_columns.numel() > max_patterns:
            counts, selected = counts.topk(max_patterns)
            unique_columns = unique_columns.index_select(0, selected)
        width = unique_columns.numel()
        retained_columns[row, :width] = unique_columns
        retained_weights[row, :width] = counts.to(torch.float32)

    bit_positions = torch.arange(
        nqubit - 1,
        -1,
        -1,
        dtype=torch.long,
        device=columns.device,
    )
    bit_matrices = ((retained_columns.unsqueeze(-1) >> bit_positions) & 1).to(
        torch.int8
    )
    return bit_matrices, retained_weights


@torch.inference_mode()
def generate_batch(
    circuit: QuantumCircuit,
    periods: Tensor,
    shifts: Tensor,
    *,
    shots: int,
    max_patterns: int,
    exact_support: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    states = periodic_state_batch(
        circuit,
        periods,
        shifts,
        exact_support=exact_support,
        device=device,
    )
    probabilities = _apply_circuit_state_batch(circuit, states).abs().pow(2)
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    columns = torch.multinomial(
        probabilities,
        shots,
        replacement=True,
        generator=generator,
    )
    bit_matrices, sample_weights = compress_measurements(
        columns,
        nqubit=circuit.num_qubits,
        max_patterns=max_patterns,
    )
    return (
        bit_matrices,
        sample_weights,
        periods.to(device=device, dtype=torch.long),
    )


def correct_counts(predicted: Tensor, periods: Tensor) -> tuple[Tensor, Tensor]:
    top1 = predicted[:, :1].eq(periods.unsqueeze(1)).any(dim=1).sum()
    top4 = predicted.eq(periods.unsqueeze(1)).any(dim=1).sum()
    return top1, top4


def reduce_totals(values: Tensor) -> Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def train_epoch(
    ddp_model: DistributedDataParallel,
    raw_model: DeepSetPeriodPredictor,
    optimizer: AdamW,
    circuit: QuantumCircuit,
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    epoch: int,
) -> tuple[float, float, float]:
    ddp_model.train()
    global_batch_size = args.local_batch_size * context.world_size
    step_count = math.ceil(args.train_draws_per_epoch / global_batch_size)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(args.seed + epoch * 1_000_003 + context.rank * 10_007)
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)

    for step in range(step_count):
        periods = training_periods(args, context, epoch=epoch, step=step)
        shifts = training_shifts(
            periods,
            seed=args.seed,
            epoch=epoch,
            step=step,
            rank=context.rank,
        )
        bit_matrices, sample_weights, period_targets = generate_batch(
            circuit,
            periods,
            shifts,
            shots=args.shots,
            max_patterns=args.max_patterns,
            exact_support=args.exact_support,
            device=context.device,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            token_logits = ddp_model(
                bit_matrices,
                sample_weights,
                target_periods=period_targets,
            )
            loss = period_token_loss(
                token_logits,
                period_targets,
                label_smoothing=args.label_smoothing,
            )
        loss.backward()
        if epoch <= args.freeze_encoder_epochs:
            clear_frozen_encoder_gradients(raw_model)
        optimizer.step()

        raw_model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            predicted, _, _ = raw_model.predict_topk_periods(
                bit_matrices,
                range(args.period_min, args.period_max + 1),
                DEFAULT_BEAM_WIDTH,
                sample_weights=sample_weights,
            )
        raw_model.train()
        top1, top4 = correct_counts(predicted, period_targets)
        batch_items = period_targets.numel()
        totals += torch.tensor(
            [float(loss.item()) * batch_items, float(top1), float(top4), batch_items],
            dtype=torch.float64,
            device=context.device,
        )

    totals = reduce_totals(totals)
    return (
        float(totals[0].item() / totals[3].item()),
        float(totals[1].item() / totals[3].item()),
        float(totals[2].item() / totals[3].item()),
    )


@torch.inference_mode()
def evaluate(
    raw_model: DeepSetPeriodPredictor,
    circuit: QuantumCircuit,
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[float, float, float]:
    raw_model.eval()
    all_periods = list(range(args.period_min, args.period_max + 1))
    if args.validation_period_limit is not None:
        all_periods = all_periods[: args.validation_period_limit]
    local_periods = all_periods[context.rank :: context.world_size]
    generator = torch.Generator(device=context.device)
    generator.manual_seed(args.seed + 900_000_007 + context.rank)
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)

    for start in range(0, len(local_periods), args.local_batch_size):
        period_values = local_periods[start : start + args.local_batch_size]
        periods = torch.tensor(period_values, dtype=torch.long)
        shifts = torch.tensor(
            [held_out_shift(period, args.seed) for period in period_values],
            dtype=torch.long,
        )
        bit_matrices, sample_weights, period_targets = generate_batch(
            circuit,
            periods,
            shifts,
            shots=args.shots,
            max_patterns=args.max_patterns,
            exact_support=args.exact_support,
            device=context.device,
            generator=generator,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pooled = raw_model.pooled_features(bit_matrices, sample_weights)
            token_logits = raw_model.decode_teacher_forced(pooled, period_targets)
            loss = period_token_loss(
                token_logits,
                period_targets,
                label_smoothing=args.label_smoothing,
            )
            predicted, _, _ = raw_model.decode_topk_from_pooled_features(
                pooled,
                DEFAULT_BEAM_WIDTH,
            )
        top1, top4 = correct_counts(predicted, period_targets)
        batch_items = period_targets.numel()
        totals += torch.tensor(
            [float(loss.item()) * batch_items, float(top1), float(top4), batch_items],
            dtype=torch.float64,
            device=context.device,
        )

    totals = reduce_totals(totals)
    raw_model.train()
    return (
        float(totals[0].item() / totals[3].item()),
        float(totals[1].item() / totals[3].item()),
        float(totals[2].item() / totals[3].item()),
    )


def checkpoint_payload(
    raw_model: DeepSetPeriodPredictor,
    optimizer: AdamW,
    args: argparse.Namespace,
    metrics: EpochMetrics,
    *,
    initialized_keys: tuple[str, ...],
    best_val_top1: float,
    stale_epochs: int,
    include_optimizer: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": {
            key: value.detach().cpu() for key, value in raw_model.state_dict().items()
        },
        "candidate_periods": list(range(args.period_min, args.period_max + 1)),
        "period_min": args.period_min,
        "period_max": args.period_max,
        "bit_width": raw_model.bit_width,
        "token_bits": TOKEN_BITS,
        "token_count": raw_model.token_count,
        "beam_width": raw_model.beam_width,
        "decoder_type": DECODER_TYPE,
        "model_architecture": raw_model.architecture,
        "num_periods": raw_model.num_periods,
        "epoch": metrics.epoch,
        "metrics": asdict(metrics),
        "best_val_top1": best_val_top1,
        "stale_epochs": stale_epochs,
        "config": serializable_config(args),
        "initialized_keys": list(initialized_keys),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "nqubit": 18,
            "top_k": DEFAULT_BEAM_WIDTH,
            "decoder_type": DECODER_TYPE,
            "token_bits": TOKEN_BITS,
        }
    )
    return config


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_history(path: Path, history: list[EpochMetrics], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": serializable_config(args),
                "history": [asdict(metrics) for metrics in history],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_resume(
    path: Path,
    raw_model: DeepSetPeriodPredictor,
    optimizer: AdamW,
) -> tuple[int, list[EpochMetrics], float, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid resume checkpoint: {path}")
    state_dict = payload.get("state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, dict) or not isinstance(optimizer_state, dict):
        raise ValueError(f"resume checkpoint lacks model/optimizer state: {path}")
    raw_model.load_state_dict(cast(dict[str, Tensor], state_dict))
    optimizer.load_state_dict(optimizer_state)
    epoch = int(payload.get("epoch", 0))
    best_top1 = float(payload.get("best_val_top1", -1.0))
    stale_epochs = int(payload.get("stale_epochs", 0))
    return epoch + 1, [], best_top1, stale_epochs


def main() -> None:
    args = parse_args()
    validate_args(args)
    context = init_distributed()
    configure_torch()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(
        args.output_dir / "train.log",
        enabled=context.is_primary,
    )

    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed + context.rank)
        circuit, nqubit = load_circuit(args.phase_path)
        model = DeepSetPeriodPredictor(
            nqubit,
            args.period_max - args.period_min + 1,
            period_min=args.period_min,
            period_max=args.period_max,
            dropout=args.dropout,
            architecture="weighted",
            beam_width=DEFAULT_BEAM_WIDTH,
        ).to(context.device)
        initialized_count, initialized_keys = initialize_encoder_from_checkpoint(
            model,
            args.init_checkpoint,
        )
        decoder_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(DECODER_PARAMETER_PREFIXES)
        ]
        encoder_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(DECODER_PARAMETER_PREFIXES)
        ]
        optimizer = AdamW(
            [
                {"params": decoder_parameters, "lr": args.learning_rate},
                {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )

        start_epoch = 1
        history: list[EpochMetrics] = []
        best_top1 = -1.0
        stale_epochs = 0
        if args.resume is not None:
            start_epoch, history, best_top1, stale_epochs = load_resume(
                args.resume,
                model,
                optimizer,
            )

        if context.is_primary:
            logger.info(
                "start world_size=%s local_batch=%s global_batch=%s train_draws=%s "
                "shots=%s max_patterns=%s parameters=%s initialized_tensors=%s config=%s",
                context.world_size,
                args.local_batch_size,
                args.local_batch_size * context.world_size,
                args.train_draws_per_epoch,
                args.shots,
                args.max_patterns,
                sum(parameter.numel() for parameter in model.parameters()),
                initialized_count,
                json.dumps(serializable_config(args), sort_keys=True),
            )

        training_start = time.perf_counter()
        for epoch in range(start_epoch, args.epochs + 1):
            if (
                context.is_primary
                and args.freeze_encoder_epochs > 0
                and epoch == args.freeze_encoder_epochs + 1
            ):
                logger.info("unfroze transferred Deep Sets encoder at epoch=%s", epoch)

            epoch_start = time.perf_counter()
            train_loss, train_top1, train_top4 = train_epoch(
                ddp_model,
                model,
                optimizer,
                circuit,
                args,
                context,
                epoch=epoch,
            )
            val_loss, val_top1, val_top4 = evaluate(
                model,
                circuit,
                args,
                context,
            )
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                train_top4=train_top4,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top4=val_top4,
                epoch_seconds=time.perf_counter() - epoch_start,
                elapsed_seconds=time.perf_counter() - training_start,
            )
            history.append(metrics)

            improved = val_top1 > best_top1
            if improved:
                best_top1 = val_top1
                stale_epochs = 0
            else:
                stale_epochs += 1

            if context.is_primary:
                if improved:
                    atomic_torch_save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            args,
                            metrics,
                            initialized_keys=initialized_keys,
                            best_val_top1=best_top1,
                            stale_epochs=stale_epochs,
                            include_optimizer=False,
                        ),
                        args.run_dir / "best.pt",
                    )
                if epoch % args.latest_interval == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            args,
                            metrics,
                            initialized_keys=initialized_keys,
                            best_val_top1=best_top1,
                            stale_epochs=stale_epochs,
                            include_optimizer=True,
                        ),
                        args.run_dir / "latest.pt",
                    )
                save_history(args.output_dir / "history.json", history, args)
                if epoch == 1 or epoch % args.log_interval == 0:
                    logger.info(
                        "epoch=%s/%s train_loss=%.6f train_top1=%.4f train_top4=%.4f "
                        "val_loss=%.6f val_top1=%.4f val_top4=%.4f epoch_seconds=%.2f "
                        "best_val_top1=%.4f stale=%s",
                        epoch,
                        args.epochs,
                        train_loss,
                        train_top1,
                        train_top4,
                        val_loss,
                        val_top1,
                        val_top4,
                        metrics.epoch_seconds,
                        best_top1,
                        stale_epochs,
                    )

            should_stop = (
                epoch >= args.min_epochs
                and stale_epochs >= args.early_stopping_patience
            )
            stop_tensor = torch.tensor(
                int(should_stop),
                dtype=torch.int32,
                device=context.device,
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                if context.is_primary:
                    logger.info(
                        "early stopping epoch=%s best_val_top1=%.4f",
                        epoch,
                        best_top1,
                    )
                break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
