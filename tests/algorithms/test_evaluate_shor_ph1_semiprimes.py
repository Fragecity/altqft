from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "algorithms" / "evaluate_shor_ph1_semiprimes.py"
    spec = importlib.util.spec_from_file_location("evaluate_shor_ph1_semiprimes", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_choose_coprime_order_finding_a_prefers_in_range_period() -> None:
    module = load_script_module()

    selection = module.choose_coprime_order_finding_a(
        2021,
        (43, 47),
        candidate_periods=tuple(range(11, 121)),
    )

    assert selection is not None
    assert selection.a == 140
    assert selection.order == 14
    assert module.factors_from_order(selection.a, 2021, selection.order) == (43, 47)


def test_order_finding_semiprimes_reports_uncovered_cases() -> None:
    module = load_script_module()

    cases, uncovered_cases = module.order_finding_semiprimes(
        15,
        15,
        candidate_periods=tuple(range(11, 121)),
    )

    assert cases == []
    assert uncovered_cases == [15]


def test_main_clips_stop_to_nqubit_state_space(monkeypatch, capsys) -> None:
    module = load_script_module()
    seen: dict[str, object] = {}

    def fake_order_finding_semiprimes(start: int, stop: int, *, candidate_periods):
        seen["start"] = start
        seen["stop"] = stop
        seen["candidate_periods"] = tuple(candidate_periods)
        return [module.SemiprimeCase(N=2047, prime_factors=(23, 89), a=3, order=94)], []

    def fake_run_shor_with_ph1(config):
        seen["config_N"] = config.N
        return SimpleNamespace(
            top1_period=None,
            predicted_period=None,
            success=False,
            factors=None,
            top_periods=(),
        )

    monkeypatch.setattr(module, "order_finding_semiprimes", fake_order_finding_semiprimes)
    monkeypatch.setattr(module, "run_shor_with_ph1", fake_run_shor_with_ph1)

    exit_code = module.main(["--start", "2045", "--stop", "2055", "--nqubit", "11"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert seen["start"] == 2045
    assert seen["stop"] == 2048
    assert seen["candidate_periods"] == tuple(range(11, 121))
    assert seen["config_N"] == 2047
    assert "requested_stop=2055" in output
    assert "effective_stop=2048" in output
    assert "state_space_limit=2048" in output
    assert "requested_N_range=2049..2055" in output
    assert "covered_cases=1" in output
