import pandas as pd
import datetime as dt
import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db
from services.price_store import upsert_bars, detect_gaps


def test_gap_detection(tmp_path):
    db.DB_PATH = str(tmp_path / "gaps.db")
    db.init_db()
    ts0 = pd.Timestamp("2024-01-01 14:30", tz="UTC")
    ts1 = pd.Timestamp("2024-01-01 15:00", tz="UTC")
    df = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]}, index=[ts0])
    upsert_bars("AAA", df)
    gaps = detect_gaps("AAA", ts0.to_pydatetime(), ts1.to_pydatetime())
    assert pd.Timestamp("2024-01-01 14:45", tz="UTC") in gaps
