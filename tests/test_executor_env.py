import importlib
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_executor_and_chunk_env(monkeypatch, caplog):
    monkeypatch.setenv("EXECUTOR_MAX_WORKERS", "5")
    monkeypatch.setenv("SCAN_EXECUTOR_MODE", "thread")
    with caplog.at_level(logging.INFO):
        ex = importlib.import_module("services.executor")
        importlib.reload(ex)
    assert ex.WORKERS == 5
    assert ex.MODE == "thread"
    assert any(
        "executor mode=thread workers=5" in rec.message for rec in caplog.records
    )

    monkeypatch.setenv("SCAN_SYMBOLS_PER_TASK", "7")
    import config

    importlib.reload(config)
    assert config.settings.scan_symbols_per_task == 7

    routes = importlib.import_module("routes")

    monkeypatch.setattr(routes, "compute_scan_for_ticker", lambda t, p: {"ticker": t})
    monkeypatch.setattr(
        routes.price_store,
        "bulk_coverage",
        lambda symbols, interval, s, e: {sym: (s, e, 10**6) for sym in symbols},
    )
    monkeypatch.setattr(routes.price_store, "covers", lambda a, b, c, d: True)
    monkeypatch.setattr(routes.settings, "scan_symbols_per_task", 7, raising=False)

    with caplog.at_level(logging.INFO):
        routes._perform_scan(["AAA"], {}, "")
    assert any("symbols_per_task=7" in rec.message for rec in caplog.records)
