import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.backfill_polygon as backfill
from services import polygon_client


def test_backfill_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    # symbols list
    symbols = ["AAA", "BBB", "CCC"]
    # checkpoint after AAA
    chk = tmp_path / "backfill_checkpoint.json"
    chk.write_text(json.dumps({"index": 1}))
    monkeypatch.setattr(backfill, "CHECKPOINT", chk)

    # stub polygon fetch
    def fake_fetch(symbols, interval, start, end):
        return {symbols[0]: pd.DataFrame()}
    monkeypatch.setattr(polygon_client, "fetch_polygon_prices", fake_fetch)

    called = []
    def fake_upsert(sym, df):
        called.append(sym)
        return 0
    monkeypatch.setattr(backfill, "upsert_bars", fake_upsert)
    monkeypatch.setattr(backfill.time, "sleep", lambda x: None)

    backfill.backfill(symbols)
    assert called == ["BBB", "CCC"]
