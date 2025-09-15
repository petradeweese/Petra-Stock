import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.backfill_polygon as backfill
from services import http_client, polygon_client


def test_backfill_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "test")
    # symbols list
    symbols = ["AAA", "BBB", "CCC"]
    # checkpoint after AAA
    chk = tmp_path / "backfill_checkpoint.json"
    chk.write_text(json.dumps({"index": 1}))
    monkeypatch.setattr(backfill, "CHECKPOINT", chk)

    # stub polygon fetch
    async def fake_fetch(symbols, interval, start, end, **kwargs):
        return {symbols[0]: pd.DataFrame()}

    monkeypatch.setattr(polygon_client, "fetch_polygon_prices_async", fake_fetch)

    called = []

    def fake_upsert(sym, df, interval="15m"):
        called.append(sym)
        return 0

    monkeypatch.setattr(backfill, "upsert_bars", fake_upsert)
    monkeypatch.setattr(backfill.time, "sleep", lambda x: None)

    client = http_client.get_client()
    backfill.backfill(symbols)
    assert client.is_closed
    assert called == ["BBB", "CCC"]
