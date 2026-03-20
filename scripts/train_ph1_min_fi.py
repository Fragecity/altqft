from __future__ import annotations

import argparse

from altqft.nn.train import TrainConfig, build_default_period_range, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 ph_1 固定 hlayout 的 min-FI 参数模型。")
    parser.add_argument("--nqubit", type=int, default=4, help="量子比特数量。")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数。")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam 学习率。")
    parser.add_argument("--seed", type=int, default=7, help="随机种子。")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="每隔多少个 epoch 记录一次日志。",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="*",
        default=None,
        help="显式指定 period_range；若省略则使用默认的稠密 period 区间。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    period_range = args.periods or build_default_period_range(args.nqubit)
    config = TrainConfig(
        nqubit=args.nqubit,
        period_range=period_range,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    artifacts = train_model(config)
    print(f"训练完成，最终 min_fi={artifacts.final_min_fi:.8f}")
    print(f"模型参数保存在: {artifacts.model_path}")
    print(f"phase 参数保存在: {artifacts.phase_path}")
    print(f"训练历史保存在: {artifacts.history_path}")
    print(f"日志保存在: {artifacts.log_path}")


if __name__ == "__main__":
    main()
