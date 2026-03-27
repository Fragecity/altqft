from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
FI_SCRIPT_DIR = ROOT_DIR / "scripts" / "fi_data_cal"
if str(FI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FI_SCRIPT_DIR))

import calculate_fi_dataset as fi_dataset
from fisher_information_utils import FiExperimentConfig


def test_should_parallelize_only_for_large_nqubit() -> None:
    assert not fi_dataset.should_parallelize(
        [FiExperimentConfig(circuit_type="qft", nqubit=9)]
    )
    assert fi_dataset.should_parallelize(
        [FiExperimentConfig(circuit_type="qft", nqubit=10)]
    )


def test_default_worker_count_cpu_uses_core_limit(monkeypatch) -> None:
    monkeypatch.setattr(fi_dataset.os, "cpu_count", lambda: 6)

    assert fi_dataset.default_worker_count(total_configs=1, device_name="cpu") == 1
    assert fi_dataset.default_worker_count(total_configs=4, device_name="cpu") == 2


def test_default_worker_count_cuda_uses_gpu_count(monkeypatch) -> None:
    monkeypatch.setattr(fi_dataset, "available_cuda_device_count", lambda: 3)

    assert fi_dataset.default_worker_count(total_configs=8, device_name="cuda") == 3


def test_worker_device_round_robins_cuda(monkeypatch) -> None:
    monkeypatch.setattr(fi_dataset, "available_cuda_device_count", lambda: 2)

    assert fi_dataset.worker_device(0, "cuda") == "cuda:0"
    assert fi_dataset.worker_device(1, "cuda") == "cuda:1"
    assert fi_dataset.worker_device(2, "cuda") == "cuda:0"
