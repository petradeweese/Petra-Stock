import time
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

import routes


app = FastAPI()
app.include_router(routes.router)


def test_scanner_progress_and_results(monkeypatch):
    monkeypatch.setattr(routes, "SP100", ["AAA", "BBB", "CCC"])

    def fake_perform_scan(tickers, params, sort_key, progress_cb=None):
        total = len(tickers)
        rows = []
        if progress_cb:
            progress_cb(0, total, "start")
        for i, t in enumerate(tickers, 1):
            time.sleep(0.01)
            if progress_cb:
                progress_cb(i, total, f"{i}/{total}")
            rows.append({
                "ticker": t,
                "direction": "UP",
                "avg_roi_pct": 1.0,
                "hit_pct": 60.0,
                "support": 10,
                "avg_tt": 1.0,
                "avg_dd_pct": 0.5,
                "stability": 0.0,
                "rule": "r1",
            })
        return rows

    monkeypatch.setattr(routes, "_perform_scan", fake_perform_scan)

    client = TestClient(app)
    res = client.post("/scanner/run", data={"scan_type": "sp100"})
    assert res.status_code == 200
    task_id = res.json()["task_id"]

    first = client.get(f"/scanner/progress/{task_id}").json()["percent"]
    assert first < 100

    final = None
    for _ in range(50):
        data = client.get(f"/scanner/progress/{task_id}").json()
        final = data["percent"]
        if data["state"] == "done":
            break
        time.sleep(0.02)
    assert data["state"] == "done"
    assert final == 100

    html = client.get(f"/scanner/results/{task_id}")
    assert html.status_code == 200
    text = html.text
    assert "AAA" in text and "BBB" in text and "CCC" in text
