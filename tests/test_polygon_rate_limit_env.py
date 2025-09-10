import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import http_client, polygon_client


def test_polygon_rate_limit_env(monkeypatch):
    captured = {}

    def fake_set_rate_limit(host, rate, capacity):
        captured["host"] = host
        captured["rate"] = rate
        captured["capacity"] = capacity

    monkeypatch.setenv("POLY_RPS", "2")
    monkeypatch.setenv("POLY_BURST", "3")
    monkeypatch.setattr(http_client, "set_rate_limit", fake_set_rate_limit)

    importlib.reload(polygon_client)

    assert captured == {"host": "api.polygon.io", "rate": 2.0, "capacity": 3}
