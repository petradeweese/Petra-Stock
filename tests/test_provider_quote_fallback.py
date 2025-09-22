import asyncio
from types import SimpleNamespace

from services import data_provider
from services.schwab_client import SchwabAPIError


def test_get_quote_falls_back_to_yfinance(monkeypatch):
    async def fake_quote(symbol, timeout_ctx=None):
        raise SchwabAPIError("boom")

    async def fake_yf(symbol):
        return {"symbol": symbol, "price": 123.45, "timestamp": None, "source": "yfinance"}

    monkeypatch.setattr(
        data_provider,
        "schwab_client",
        SimpleNamespace(get_quote=fake_quote, get_price_history=lambda *a, **k: None),
    )
    monkeypatch.setattr(data_provider, "_fetch_yfinance_quote", fake_yf)

    quote = asyncio.run(data_provider.get_quote_async("AAA"))
    assert quote["source"] == "yfinance"
    assert quote["price"] == 123.45
