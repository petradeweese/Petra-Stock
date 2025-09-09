import sqlite3
import sys
from pathlib import Path

import pandas as pd
from starlette.requests import Request
from pytest import approx

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
        "VALUES ('AAA','UP','15m','r1',1.0,1.0,1,'Hours')"
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
    df_create = pd.DataFrame({"Close": [100.0]}, index=[ts0])
    df_update = pd.DataFrame({"Close": [100.0, 102.0]}, index=[ts0, ts1])
    calls = {"n": 0}

    def fake_fetch_prices(tickers, interval, lookback_years):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {tickers[0]: df_create}
        return {tickers[0]: df_update}

    monkeypatch.setattr(routes, "fetch_prices", fake_fetch_prices)

    class DummyResponse:
        def __init__(self, name, context):
            self.template = type("T", (), {"name": name})
            self.context = context

    def dummy_template_response(name, context):
        return DummyResponse(name, context)

    monkeypatch.setattr(routes.templates, "TemplateResponse", dummy_template_response)

    request = Request({"type": "http"})
    forward_page(request, db=cur)
    cur.execute("SELECT roi_pct, status FROM forward_tests")
    row = cur.fetchone()
    assert row["roi_pct"] == 0.0
    assert row["status"] == "OPEN"

    forward_page(request, db=cur)
    cur.execute("SELECT roi_pct, status, hit_pct, dd_pct FROM forward_tests")
    row = cur.fetchone()
    assert row["roi_pct"] == approx(2.0)
    assert row["status"] == "HIT"
    assert row["hit_pct"] == 100.0
    assert row["dd_pct"] == 0.0
    conn.close()
