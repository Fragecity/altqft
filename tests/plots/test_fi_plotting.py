from __future__ import annotations

import json
from pathlib import Path

import pytest

from altqft.plotting.fi import (
    FiResultRecord,
    PlotData,
    load_hp1_shared_results,
    load_results,
    plot_fi_dataset,
)


def test_load_results_missing_input_returns_empty_list(tmp_path: Path) -> None:
    assert load_results(tmp_path / "missing.pkl") == []


def test_load_hp1_shared_results_prefers_recomputed_min_fi(tmp_path: Path) -> None:
    summary_path = tmp_path / "hp1_shared_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "nqubit": 5,
                        "min_fi": 7.5,
                        "best_epoch": {
                            "epoch": 3,
                            "loss": -100.0,
                            "min_fi": 1.25,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_hp1_shared_results(summary_path)

    assert len(records) == 1
    assert records[0].circuit_type == "HP1_shared"
    assert records[0].nqubit == 5
    assert records[0].fi_value == 7.5


def test_plot_fi_dataset_does_not_load_optimized_ph1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[FiResultRecord] = []

    monkeypatch.setattr(
        "altqft.plotting.fi.load_results",
        lambda input_path: [
            FiResultRecord("ph1", 7, 1.0),
            FiResultRecord("HP1_random", 7, 100.0),
        ],
    )
    monkeypatch.setattr(
        "altqft.plotting.fi.load_optimized_ph1_results",
        lambda input_path: [FiResultRecord("ph1_optimized", 7, 100.0)],
    )
    monkeypatch.setattr(
        "altqft.plotting.fi.load_hp1_shared_results",
        lambda input_path: [FiResultRecord("HP1_shared", 7, 2.0)],
    )

    def fake_plot(data_dict: PlotData, output_path: Path) -> None:
        del output_path
        for circuit_type, x_y_dict in data_dict.items():
            for nqubit, values in x_y_dict.items():
                for value in values:
                    captured.append(FiResultRecord(circuit_type, nqubit, value))

    monkeypatch.setattr("altqft.plotting.fi.plot_fi_vs_nqubits", fake_plot)

    plot_fi_dataset(
        input_path=tmp_path / "fi.pkl",
        optimized_summary_path=tmp_path / "optimized.json",
        hp1_shared_summary_path=tmp_path / "hp1_shared.json",
        output_dir=tmp_path,
    )

    assert {record.circuit_type for record in captured} == {"ph1", "HP1_shared"}
