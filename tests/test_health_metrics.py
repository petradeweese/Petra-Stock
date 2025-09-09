import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from routes import healthz, metrics


def test_health_endpoint():
    assert healthz() == {"status": "ok"}


def test_metrics_endpoint():
    resp = metrics()
    assert resp.media_type.startswith("text/plain")
    assert b"scan_duration_seconds" in resp.body
