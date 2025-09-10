import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from services import polygon_client
from services.price_store import clear_cache, detect_gaps, upsert_bars


def test_gap_fill_fetch(monkeypatch, tmp_path):
    db.DB_PATH = str(tmp_path / "gapfill.db")
    db.init_db()
    start = pd.Timestamp("2024-01-01 14:30", tz="UTC")
    missing = pd.Timestamp("2024-01-01 14:45", tz="UTC")
    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=[start],
    )
    upsert_bars("AAA", df)

    gaps = detect_gaps(
        "AAA",
        start.to_pydatetime(),
        (missing + pd.Timedelta(minutes=15)).to_pydatetime(),
    )
    assert missing in gaps

    def fake_fetch(symbols, interval, start_dt, end_dt):
        return {
            "AAA": pd.DataFrame(
                {"Open": [2], "High": [2], "Low": [2], "Close": [2], "Volume": [2]},
                index=[missing],
            )
        }

    monkeypatch.setattr(polygon_client, "fetch_polygon_prices", fake_fetch)
    fetched = polygon_client.fetch_polygon_prices(
        ["AAA"],
        "15m",
        start.to_pydatetime(),
        (missing + pd.Timedelta(minutes=15)).to_pydatetime(),
    )["AAA"]
    upsert_bars("AAA", fetched)
    clear_cache()

    gaps = detect_gaps(
        "AAA",
        start.to_pydatetime(),
        (missing + pd.Timedelta(minutes=15)).to_pydatetime(),
    )
    assert gaps == []
