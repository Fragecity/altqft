from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CONFIG_PATTERN = re.compile(r"config=(\{.*\})")
EPOCH_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)/(?P<total>\d+)\s+loss=(?P<loss>-?\d+(?:\.\d+)?)\s+min_fi=(?P<min_fi>-?\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporary helper to convert ph1_min_fi log to history json."
    )
    parser.add_argument("log_file", type=Path, help="Path to training log file.")
    parser.add_argument("output_file", type=Path, help="Path to output history json file.")
    return parser.parse_args()


def parse_log(log_file: Path) -> dict[str, Any]:
    config: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []

    for line in log_file.read_text(encoding="utf-8").splitlines():
        if config is None:
            match = CONFIG_PATTERN.search(line)
            if match:
                config = json.loads(match.group(1))
                continue

        match = EPOCH_PATTERN.search(line)
        if not match:
            continue

        history.append(
            {
                "epoch": int(match.group("epoch")),
                "loss": float(match.group("loss")),
                "min_fi": float(match.group("min_fi")),
            }
        )

    if config is None:
        raise ValueError(f"Cannot find config payload in {log_file}")

    return {"config": config, "history": history}


def main() -> None:
    args = parse_args()
    payload = parse_log(args.log_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
