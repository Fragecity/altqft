from __future__ import annotations

import altqft.fi.dataset as fi_dataset
from altqft.fi.dataset import FiExperimentConfig
import pytest


def test_should_parallelize_only_for_large_nqubit() -> None:
    assert not fi_dataset.should_parallelize(
        [FiExperimentConfig(circuit_type="qft", nqubit=9)]
    )
    assert fi_dataset.should_parallelize(
        [FiExperimentConfig(circuit_type="qft", nqubit=10)]
    )


def test_default_worker_count_cpu_uses_core_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_dataset.os, "cpu_count", lambda: 6)

    assert fi_dataset.default_worker_count(total_configs=1, device_name="cpu") == 1
    assert fi_dataset.default_worker_count(total_configs=4, device_name="cpu") == 2


def test_default_worker_count_cuda_uses_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_dataset, "available_cuda_device_count", lambda: 3)

    assert fi_dataset.default_worker_count(total_configs=8, device_name="cuda") == 3


def test_worker_device_round_robins_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_dataset, "available_cuda_device_count", lambda: 2)

    assert fi_dataset.worker_device(0, "cuda") == "cuda:0"
    assert fi_dataset.worker_device(1, "cuda") == "cuda:1"
    assert fi_dataset.worker_device(2, "cuda") == "cuda:0"


def test_build_configs_includes_selected_non_optimized_fi_curves() -> None:
    configs = fi_dataset.build_configs(range(7, 8))
    config_keys = {
        (config.circuit_type, config.nlayer, config.repeat)
        for config in configs
    }

    assert ("qft", None, 1) in config_keys
    assert ("ph1", None, 1) in config_keys
    assert ("HP1_random", None, fi_dataset.SAMPLES) not in config_keys
    assert ("HPrandom", 1, fi_dataset.SAMPLES) in config_keys
    assert ("HPrandom_phase", 6, fi_dataset.SAMPLES) in config_keys
