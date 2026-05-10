from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


def load_train_eta_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "tools" / "train_eta.py"
    spec = importlib.util.spec_from_file_location("train_eta", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_estimate_training_eta_uses_recent_intervals(tmp_path: Path) -> None:
    module = load_train_eta_module()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-04-01 14:48:50,229 | INFO | epoch=1/300 train_loss=4.6",
                "2026-04-01 15:05:45,886 | INFO | epoch=10/300 train_loss=4.4",
                "2026-04-01 15:24:47,637 | INFO | epoch=20/300 train_loss=3.4",
                "2026-04-01 15:43:49,703 | INFO | epoch=30/300 train_loss=2.0",
            ]
        ),
        encoding="utf-8",
    )

    records = module.parse_epoch_records(log_path)
    estimate = module.estimate_training_eta(records, window=2)

    assert len(records) == 4
    assert estimate.last_epoch == 30
    assert estimate.total_epochs == 300
    assert estimate.remaining_epochs == 270
    assert estimate.intervals_used == 2
    assert abs(estimate.seconds_per_epoch - 114.19085) < 0.01
    assert estimate.estimated_completion == datetime(2026, 4, 2, 0, 17, 41, 232500)
