import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import routes
import scanner


def test_perform_scan_counts_skipped(monkeypatch):
    tickers = ["AAA", "BBB"]

    def fake_scan(ticker, params):
        if ticker == "BBB":
            return None
        return {"ticker": ticker}

    monkeypatch.setattr(routes, "compute_scan_for_ticker", fake_scan)
    monkeypatch.setattr(routes, "preload_prices", lambda *a, **k: None)

    rows, skipped = routes._perform_scan(tickers, {}, "")
    assert rows == [{"ticker": "AAA"}]
    assert skipped == 1


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
