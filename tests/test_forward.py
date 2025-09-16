import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from starlette.requests import Request
from pytest import approx

os.environ["SCAN_EXECUTOR_MODE"] = "thread"
sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from routes import forward_page
import routes


def test_forward_tracking_only_future_bars(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule, target_pct, stop_pct, window_value, window_unit) "
        "VALUES ('AAA','UP','15m','r1',2.0,1.0,1,'Hours')"
    )
    conn.commit()

    # compute_scan_for_ticker should not run
    monkeypatch.setattr(
        routes,
        "compute_scan_for_ticker",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan should not run")),
    )

    ts0 = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 10:15:00", tz="UTC")
    df_create = pd.DataFrame(
        {"Open": [100.0], "Close": [100.0], "High": [100.0], "Low": [100.0]},
        index=[ts0],
    )
    df_update = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "Close": [100.0, 102.0],
            "High": [100.0, 103.0],
            "Low": [100.0, 97.0],
        },
        index=[ts0, ts1],
    )
    calls = {"n": 0}

    def fake_get_prices(tickers, interval, start, end):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {tickers[0]: df_create}
        return {tickers[0]: df_update}

    monkeypatch.setattr(routes, "get_prices", fake_get_prices)

    class DummyResponse:
        def __init__(self, name, context):
            self.template = type("T", (), {"name": name})
            self.context = context

    def dummy_template_response(request, name, context):
        return DummyResponse(name, context)

    monkeypatch.setattr(routes.templates, "TemplateResponse", dummy_template_response)

    request = Request({"type": "http"})
    forward_page(request, db=cur)
    cur.execute("SELECT roi_forward, status FROM forward_tests")
    row = cur.fetchone()
    assert row["roi_forward"] == 0.0
    assert row["status"] == "queued"

    forward_page(request, db=cur)
    cur.execute(
        "SELECT roi_forward, option_roi_proxy, status, hit_forward, dd_forward, roi_1, mae, mfe, time_to_stop, exit_reason, bars_to_exit, max_drawdown_pct, max_runup_pct, r_multiple FROM forward_tests"
    )
    row = cur.fetchone()
    assert row["roi_forward"] == approx(-1.1582733813, rel=1e-3)
    assert row["option_roi_proxy"] == approx(-2.8956834532, rel=1e-3)
    assert row["status"] == "ok"
    assert row["hit_forward"] == 0.0
    assert row["exit_reason"] == "stop"
    assert row["bars_to_exit"] == 1
    assert row["dd_forward"] == approx(3.1550759392, rel=1e-3)
    assert row["max_drawdown_pct"] == approx(3.1550759392, rel=1e-3)
    assert row["max_runup_pct"] == approx(2.8353317346, rel=1e-3)
    assert row["roi_1"] == approx(-1.1582733813, rel=1e-3)
    assert row["mae"] == approx(-3.1550759392, rel=1e-3)
    assert row["mfe"] == approx(2.8353317346, rel=1e-3)
    assert row["time_to_stop"] == approx(0.0)
    assert row["r_multiple"] == approx(-1.0)
    conn.close()
