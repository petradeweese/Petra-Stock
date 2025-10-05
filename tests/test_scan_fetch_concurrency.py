import asyncio
from concurrent.futures import Future
from datetime import datetime, timedelta


def test_scan_fetch_concurrency_clamped(monkeypatch):
    from routes import _perform_scan
    from routes import executor, settings

    tickers = ["AAA"]
    params = {"interval": "15m", "lookback_years": 1.0}

    # Clamp configuration to zero to exercise the guard path.
    monkeypatch.setattr(settings, "scan_fetch_concurrency", 0)

    # Simplify coverage to force a fetch for every ticker.
    monkeypatch.setattr(
        "routes.price_store.bulk_coverage", lambda symbols, *_args, **_kwargs: {}
    )
    monkeypatch.setattr("routes.price_store.covers", lambda *a, **k: False)
    monkeypatch.setattr("routes.expected_bar_count", lambda *a, **k: 1)

    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1)
    monkeypatch.setattr("routes.window_from_lookback", lambda *_a: (start, end))

    async def _to_thread(fn, *args, **kwargs):
        fn(*args, **kwargs)

    monkeypatch.setattr("routes.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("routes.fetch_prices", lambda *a, **k: {symbol: {} for symbol in a[0]})

    # Ensure the semaphore is created with a positive value.
    original_semaphore = asyncio.Semaphore

    def _checking_semaphore(value):
        assert value >= 1
        return original_semaphore(value)

    monkeypatch.setattr("routes.asyncio.Semaphore", _checking_semaphore)

    def _scan_chunk(symbols, _params):
        rows = []
        for sym in symbols:
            rows.append(
                (
                    sym,
                    {
                        "ticker": sym,
                        "avg_roi_pct": 1.0,
                        "hit_pct": 50.0,
                        "support": 10,
                        "hit_lb95": 0.5,
                    },
                    0.01,
                )
            )
        return rows

    monkeypatch.setattr("routes._scan_chunk", _scan_chunk)

    class _DummyExecutor:
        def submit(self, fn, chunk, params):
            fut: Future = Future()
            fut.set_result(fn(chunk, params))
            return fut

    monkeypatch.setattr(executor, "EXECUTOR", _DummyExecutor())
    monkeypatch.setattr(executor, "MODE", "thread")
    monkeypatch.setattr(executor, "WORKERS", 1)

    rows, skipped, metrics = _perform_scan(tickers, params, sort_key="ticker")

    assert skipped == 0
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAA"
    # The guard still executes the fetch block, so metrics are populated.
    assert metrics["symbols_gap"] == 1

