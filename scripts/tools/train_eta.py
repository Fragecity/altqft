from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| "
    r"epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+)\b"
)
DEFAULT_LOG_PATH = Path("outputs/period_recovery_10q.log")
DEFAULT_WINDOW = 5


@dataclass(frozen=True, slots=True)
class EpochRecord:
    timestamp: datetime
    epoch: int
    total_epochs: int


@dataclass(frozen=True, slots=True)
class EtaEstimate:
    last_epoch: int
    total_epochs: int
    remaining_epochs: int
    intervals_used: int
    seconds_per_epoch: float
    estimated_completion: datetime
    remaining_duration: timedelta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate training completion time from an epoch log.",
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Training log path. Defaults to outputs/period_recovery_10q.log.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Use the most recent N epoch intervals for the estimate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the estimate as JSON.",
    )
    return parser.parse_args()


def parse_epoch_records(log_path: Path) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.match(line.strip())
        if match is None:
            continue
        records.append(
            EpochRecord(
                timestamp=datetime.strptime(
                    match.group("timestamp"),
                    "%Y-%m-%d %H:%M:%S,%f",
                ),
                epoch=int(match.group("epoch")),
                total_epochs=int(match.group("total_epochs")),
            )
        )
    return records


def estimate_training_eta(
    records: list[EpochRecord],
    *,
    window: int,
) -> EtaEstimate:
    if window < 1:
        raise ValueError("window must be positive")
    if len(records) < 2:
        raise ValueError("need at least two logged epochs to estimate ETA")

    last_record = records[-1]
    intervals: list[tuple[float, int]] = []
    for previous, current in zip(records, records[1:]):
        epoch_delta = current.epoch - previous.epoch
        seconds_delta = (current.timestamp - previous.timestamp).total_seconds()
        if epoch_delta <= 0 or seconds_delta <= 0:
            continue
        intervals.append((seconds_delta, epoch_delta))

    if not intervals:
        raise ValueError("log does not contain any valid positive epoch intervals")

    selected_intervals = intervals[-window:]
    total_seconds = sum(seconds for seconds, _ in selected_intervals)
    total_epochs = sum(epoch_delta for _, epoch_delta in selected_intervals)
    seconds_per_epoch = total_seconds / float(total_epochs)
    remaining_epochs = max(0, last_record.total_epochs - last_record.epoch)
    remaining_duration = timedelta(seconds=remaining_epochs * seconds_per_epoch)
    estimated_completion = last_record.timestamp + remaining_duration

    return EtaEstimate(
        last_epoch=last_record.epoch,
        total_epochs=last_record.total_epochs,
        remaining_epochs=remaining_epochs,
        intervals_used=len(selected_intervals),
        seconds_per_epoch=seconds_per_epoch,
        estimated_completion=estimated_completion,
        remaining_duration=remaining_duration,
    )


def format_duration(duration: timedelta) -> str:
    total_seconds = int(round(duration.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def main() -> None:
    args = parse_args()
    records = parse_epoch_records(args.log_path)
    estimate = estimate_training_eta(records, window=args.window)

    if args.json:
        payload = asdict(estimate)
        payload["estimated_completion"] = estimate.estimated_completion.isoformat(sep=" ")
        payload["remaining_duration"] = format_duration(estimate.remaining_duration)
        print(json.dumps(payload, indent=2))
        return

    print(f"log_path={args.log_path}")
    print(
        f"progress={estimate.last_epoch}/{estimate.total_epochs} "
        f"remaining_epochs={estimate.remaining_epochs}"
    )
    print(
        f"intervals_used={estimate.intervals_used} "
        f"seconds_per_epoch={estimate.seconds_per_epoch:.2f}"
    )
    print(f"remaining={format_duration(estimate.remaining_duration)}")
    print(
        "estimated_completion="
        f"{estimate.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
