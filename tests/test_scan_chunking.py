import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.mark.slow
def test_chunking_reduces_futures(monkeypatch):
    tickers = [f"T{i}" for i in range(9)]

    routes = importlib.import_module("routes")
    executor = importlib.import_module("services.executor")

    monkeypatch.setattr(routes, "compute_scan_for_ticker", lambda t, p: {"ticker": t})
    monkeypatch.setattr(
        routes.price_store,
        "bulk_coverage",
        lambda symbols, interval, s, e: {sym: (s, e, 10**6) for sym in symbols},
    )
    monkeypatch.setattr(routes.price_store, "covers", lambda a, b, c, d: True)
    monkeypatch.setattr(routes.settings, "scan_symbols_per_task", 3, raising=False)

    submissions = []
    orig_submit = executor.EXECUTOR.submit

    def submit(fn, *args, **kwargs):
        submissions.append(1)
        return orig_submit(fn, *args, **kwargs)

    monkeypatch.setattr(executor.EXECUTOR, "submit", submit)

    rows, skipped, _ = routes._perform_scan(tickers, {}, "")
    assert {r["ticker"] for r in rows} == set(tickers)
    assert skipped == 0
    assert len(submissions) == math.ceil(len(tickers) / 3)
