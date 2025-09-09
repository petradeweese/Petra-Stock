import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import routes


def test_scanner_run_parallel_handles_errors(monkeypatch, caplog):
    monkeypatch.setenv("SCAN_WORKERS", "2")

    tickers = ["AAA", "BAD", "CCC"]

    def fake_scan(ticker, params):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return {
            "ticker": ticker,
            "direction": "UP",
            "avg_roi_pct": 1.0,
            "hit_pct": 60.0,
            "support": 10,
            "avg_tt": 1.0,
            "avg_dd_pct": 0.5,
            "stability": 0.0,
            "rule": "r1",
        }

    monkeypatch.setattr(routes, "compute_scan_for_ticker", fake_scan)

    with caplog.at_level(logging.ERROR):
        rows = routes._perform_scan(tickers, {}, "")

    assert {r["ticker"] for r in rows} == {"AAA", "CCC"}
    assert any("BAD" in rec.message for rec in caplog.records)
