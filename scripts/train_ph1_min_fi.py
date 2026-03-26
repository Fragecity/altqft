from __future__ import annotations

import argparse

from altqft.nn.train import TrainConfig, build_default_period_range, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PH-1 minimum-FI model.")
    parser.add_argument("--nqubit", type=int, default=4, help="Number of qubits.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Log every N epochs.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="*",
        default=None,
        help="Explicit period range. Defaults to the built-in heuristic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    period_range = (
        build_default_period_range(args.nqubit)
        if args.periods is None
        else list(args.periods)
    )
    config = TrainConfig(
        nqubit=args.nqubit,
        period_range=period_range,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    artifacts = train_model(config)
    print(f"final_min_fi={artifacts.final_min_fi:.8f}")
    print(f"model_path={artifacts.model_path}")
    print(f"phase_path={artifacts.phase_path}")
    print(f"history_path={artifacts.history_path}")
    print(f"log_path={artifacts.log_path}")
