import app


def test_compute_scan_for_ticker_selects_best(monkeypatch):
    def fake_scan(ticker, params):
        if params["direction"] == "UP":
            return {"ticker": ticker, "direction": "UP", "avg_roi_pct": 1,
                    "hit_pct": 60, "support": 5, "stability": 1}
        return {"ticker": ticker, "direction": "DOWN", "avg_roi_pct": 2,
                "hit_pct": 70, "support": 10, "stability": 2}

    monkeypatch.setattr(app, "_desktop_like_single", fake_scan)
    params = {"direction": "BOTH"}
    result = app.compute_scan_for_ticker("ABC", params)
    assert result["direction"] == "DOWN"
    assert result["avg_roi_pct"] == 2
