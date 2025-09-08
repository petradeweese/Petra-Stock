import sqlite3
import sys
from pathlib import Path

from starlette.requests import Request

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from routes import forward_page
import routes


def test_forward_page_runs_scan_and_records(tmp_path, monkeypatch):
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO favorites(ticker, direction, interval, rule) VALUES ('AAA','UP','15m','r1')"
    )
    conn.commit()

    def fake_scan(ticker, params):
        return {"avg_roi_pct": 0.5, "hit_pct": 60.0, "avg_dd_pct": 0.1}

    monkeypatch.setattr(routes, "compute_scan_for_ticker", fake_scan)

    class DummyResponse:
        def __init__(self, name, context):
            self.template = type("T", (), {"name": name})
            self.context = context

    def dummy_template_response(name, context):
        return DummyResponse(name, context)

    monkeypatch.setattr(routes.templates, "TemplateResponse", dummy_template_response)

    request = Request({"type": "http"})
    resp = forward_page(request, db=cur)
    assert resp.template.name == "forward.html"
    cur.execute("SELECT COUNT(*) FROM forward_tests")
    assert cur.fetchone()[0] == 1
    conn.close()
