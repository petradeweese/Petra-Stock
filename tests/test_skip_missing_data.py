import logging
import os
import sys
from pathlib import Path

import pandas as pd

os.environ["SCAN_EXECUTOR_MODE"] = "thread"
sys.path.append(str(Path(__file__).resolve().parents[1]))

import routes
import scanner
from services import market_data


def test_perform_scan_counts_skipped(monkeypatch):
    tickers = ["AAA", "BBB"]

    def fake_scan(ticker, params):
        if ticker == "BBB":
            return None
        return {"ticker": ticker}

    monkeypatch.setattr(routes, "compute_scan_for_ticker", fake_scan)
    monkeypatch.setattr(
        routes.price_store,
        "bulk_coverage",
        lambda symbols, interval, s, e: {sym: (s, e, 10**6) for sym in symbols},
    )
    monkeypatch.setattr(routes.price_store, "covers", lambda a, b, c, d: True)

    rows, skipped, metrics = routes._perform_scan(tickers, {}, "")
    assert rows == [{"ticker": "AAA"}]
    assert skipped == 1
    assert metrics["symbols_no_gap"] == 2


def test_compute_scan_skips_when_no_data(monkeypatch, caplog):
    def fake_fetch(symbols, interval, lookback, provider=None):
        return {symbols[0]: pd.DataFrame()}

    monkeypatch.setattr(scanner, "fetch_prices", fake_fetch)
    monkeypatch.setattr(scanner, "_PRICE_DATA", {})

    params = {"interval": "1d", "lookback_years": 1.0}

    with caplog.at_level(logging.INFO):
        res = scanner.compute_scan_for_ticker("XYZ", params)

    assert res is None
    assert any("skip_no_data symbol=XYZ" in rec.message for rec in caplog.records)


def test_ensure_coverage_intraday(monkeypatch):
    import datetime as dt

    start = dt.datetime(2024, 1, 8, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 2, 15, tzinfo=dt.timezone.utc)
    interval = "1h"
    expected = market_data.expected_bar_count(start, end, interval)
    df = pd.DataFrame({"close": range(expected)})

    monkeypatch.setattr(scanner, "_PRICE_DATA", {})

    def fake_fetch(symbols, interval, lookback, provider=None):
        return {symbols[0]: df}

    monkeypatch.setattr(scanner, "fetch_prices", fake_fetch)
    monkeypatch.setattr(scanner, "window_from_lookback", lambda _lb: (start, end))

    assert scanner._ensure_coverage("XYZ", interval, 0.1)
