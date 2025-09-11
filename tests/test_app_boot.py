import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

sys.path.append(str(Path(__file__).resolve().parents[1]))

import db


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "app_module", Path(__file__).resolve().parents[1] / "app.py"
    )
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    return app_module


def test_app_fails_if_migrations_fail(monkeypatch):
    def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "init_db", fail)
    with pytest.raises(RuntimeError):
        _load_app()


def test_app_boots_when_migrations_disabled(monkeypatch):
    def fail():
        raise RuntimeError("boom")

    monkeypatch.setenv("RUN_MIGRATIONS", "0")
    monkeypatch.setattr(db, "init_db", fail)

    app_module = _load_app()
    assert isinstance(app_module.app, FastAPI)
