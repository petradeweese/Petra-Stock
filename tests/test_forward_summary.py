import math

import pytest

from services import forward_summary


def test_forward_summary_cache(monkeypatch):
    forward_summary._summary_cache.clear()
    events = []
    monkeypatch.setattr(forward_summary, "log_telemetry", events.append)

    ticker_rows = [
        {
            "outcome": "hit",
            "roi": 0.02,
            "tt_bars": 5,
            "dd": 0.003,
            "exit_ts": "2025-09-16T13:15:00Z",
        }
    ]

    monotonic_values = [100.0]
    monkeypatch.setattr(
        forward_summary.time,
        "monotonic",
        lambda: monotonic_values[0],
    )

    summary_first = forward_summary.get_cached_forward_summary("fav-1", ticker_rows)
    assert summary_first["n"] == 1
    assert events[-1]["event"] == "forward_summary_refresh"

    summary_cached = forward_summary.get_cached_forward_summary("fav-1", [])
    assert summary_cached == summary_first
    assert events[-1]["event"] == "forward_summary_cached"

    monotonic_values[0] += forward_summary.SUMMARY_CACHE_TTL_SECONDS + 1
    newer_rows = ticker_rows + [
        {
            "outcome": "stop",
            "roi": -0.01,
            "tt_bars": 4,
            "dd": 0.002,
            "exit_ts": "2025-09-17T13:15:00Z",
        }
    ]
    summary_refreshed = forward_summary.get_cached_forward_summary("fav-1", newer_rows)
    assert summary_refreshed["n"] == 2
    assert summary_refreshed["hits"] == 1
    assert events[-1]["event"] == "forward_summary_refresh"

    forward_summary.invalidate_forward_summary("fav-1")
    post_invalidate = forward_summary.get_cached_forward_summary("fav-1", [])
    assert post_invalidate["n"] == 0
    assert post_invalidate["hits"] == 0


@pytest.mark.parametrize(
    "roi, expected",
    [
        (float("nan"), None),
        (None, None),
        (0.01, 0.01),
    ],
)
def test_forward_summary_sanitizes_optional(roi, expected):
    rows = [
        {
            "outcome": "hit",
            "roi": roi,
            "tt_bars": None,
            "dd": None,
            "exit_ts": "2025-09-16T00:00:00Z",
        }
    ]
    summary = forward_summary.compute_forward_summary(rows)
    assert summary["avg_roi"] == expected
    assert summary["hit_rate"] == 1.0
    assert math.isfinite(summary["hit_lb95"])
