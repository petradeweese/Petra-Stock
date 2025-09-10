import datetime as dt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import market_data


def test_provider_switching(monkeypatch):
    calls = {"y": 0, "p": 0, "d": 0}

    def fake_yahoo(symbols, interval, lookback):
        calls["y"] += 1
        return {}

    def fake_polygon(symbols, interval, start, end):
        calls["p"] += 1
        return {}

    def fake_db(symbols, start, end):
        calls["d"] += 1
        return {}

    monkeypatch.setattr(market_data, "yahoo_fetch", fake_yahoo)
    monkeypatch.setattr(market_data, "fetch_polygon_prices", fake_polygon)
    monkeypatch.setattr(market_data, "get_prices_from_db", fake_db)

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)

    market_data.get_prices(["AAA"], "15m", start, end, provider="yahoo")
    market_data.get_prices(["AAA"], "15m", start, end, provider="polygon")
    market_data.get_prices(["AAA"], "15m", start, end, provider="db")

    assert calls == {"y": 1, "p": 1, "d": 1}
