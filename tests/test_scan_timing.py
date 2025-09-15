import os
import sys
import time
from pathlib import Path

import pytest

os.environ["SCAN_EXECUTOR_MODE"] = "thread"
sys.path.append(str(Path(__file__).resolve().parents[1]))

import routes


@pytest.mark.slow
def test_scan_completes_quickly(monkeypatch):
    tickers = [f"T{i}" for i in range(50)]

    def fake_scan(t, params):
        time.sleep(0.01)
        return {"ticker": t}

    monkeypatch.setattr(routes, "compute_scan_for_ticker", fake_scan)
    monkeypatch.setattr(
        routes.price_store,
        "bulk_coverage",
        lambda symbols, interval, s, e: {sym: (s, e, 10**6) for sym in symbols},
    )
    monkeypatch.setattr(routes.price_store, "covers", lambda a, b, c, d: True)

    start = time.perf_counter()
    rows, skipped, metrics = routes._perform_scan(tickers, {}, "")
    duration = time.perf_counter() - start
    assert len(rows) == 50
    assert skipped == 0
    assert duration < 2.0
