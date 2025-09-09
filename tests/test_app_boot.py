import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db


def test_app_boots_even_if_migrations_fail(monkeypatch):
    def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "init_db", fail)

    spec = importlib.util.spec_from_file_location(
        "app_module", Path(__file__).resolve().parents[1] / "app.py"
    )
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)

    assert isinstance(app_module.app, FastAPI)
