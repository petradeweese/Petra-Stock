"""Helpers for building synthetic favorites events in development."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class SimEvent:
    symbol: str
    direction: str
    outcome: str
    bar_ts: datetime
    note: str = "simulated"


def _normalize(symbol: str) -> str:
    return (symbol or "").upper()


def build_sim_hit(symbol: str, direction: str, bar_ts: datetime) -> SimEvent:
    return SimEvent(symbol=_normalize(symbol), direction=(direction or "UP").upper(), outcome="hit", bar_ts=bar_ts)


def build_sim_stop(symbol: str, direction: str, bar_ts: datetime) -> SimEvent:
    return SimEvent(symbol=_normalize(symbol), direction=(direction or "UP").upper(), outcome="stop", bar_ts=bar_ts)


__all__ = ["SimEvent", "build_sim_hit", "build_sim_stop"]
