from __future__ import annotations

from altqft.plotting.period_recovery import build_metric_series


def test_build_metric_series_uses_configured_top_k_suffix() -> None:
    payload = {
        "config": {"top_k": 10},
        "history": [
            {
                "epoch": 1,
                "val_loss": 6.4,
                "val_top1": 0.1,
                "val_top10": 0.3,
            },
            {
                "epoch": 2,
                "val_loss": 5.9,
                "val_top1": 0.2,
                "val_top10": 0.4,
            },
        ],
    }

    epochs, losses, top1_values, topk_values, top_k = build_metric_series(
        payload,
        "val",
    )

    assert epochs == [1, 2]
    assert losses == [6.4, 5.9]
    assert top1_values == [0.1, 0.2]
    assert topk_values == [0.3, 0.4]
    assert top_k == 10
