from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Iterable
from typing import Any

import pytest
from qiskit import QuantumCircuit

from altqft.nn.train import EpochResult, TrainConfig


def _load_sweep_module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "train" / "train_hp1_shared_sweep.py"
    spec = importlib.util.spec_from_file_location("train_hp1_shared_sweep_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_entry_uses_recomputed_min_fi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep = _load_sweep_module()
    config = TrainConfig(
        nqubit=4,
        period_range=[2, 3],
        model_dir=tmp_path / "model",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        model_stem="hp1_shared_unit",
        objective="hp1_shared_fi_shift",
        ansatz="HP1_shared",
        train_device="cpu",
    )
    config.model_dir.mkdir(parents=True)
    config.phase_path.write_text(
        json.dumps({"phases": [0.1, 0.2]}),
        encoding="utf-8",
    )

    def fake_min_fi(
        circuit: QuantumCircuit,
        period_range: Iterable[int],
        device: str | None = None,
    ) -> float:
        del circuit, period_range, device
        return 12.5

    monkeypatch.setattr(sweep, "min_fi", fake_min_fi)

    entry = sweep.build_summary_entry(
        config,
        [
            EpochResult(epoch=1, loss=-1.0, min_fi=1.0),
            EpochResult(epoch=2, loss=-2.0, min_fi=2.0),
        ],
    )

    assert entry["min_fi"] == 12.5
    assert entry["min_fi_source"] == sweep.MIN_FI_SOURCE
    assert entry["best_epoch"]["loss"] == -2.0
    assert entry["best_epoch"]["min_fi"] == 2.0
