"""Utilities for aggregating forward test results into cohorts.

This module provides a helper used by forward-test monitoring features.
The ``cohort_rollup`` function groups rows by week and strategy and computes
rolling metrics such as average ROI, hit rate and drawdown.  Each cohort is
assigned a colour-coded status based on configurable thresholds so callers can
quickly highlight deteriorating performance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class CohortRow:
    """Internal representation of a rollup row."""

    week: pd.Timestamp
    rule: str
    count: int
    hit_pct: float | None
    roi: float | None
    dd: float | None
    status: str


def cohort_rollup(
    rows: Iterable[Dict],
    *,
    rolling_n: int = 20,
    hit_red: float = 55.0,
    dd_red: float = 60.0,
) -> List[Dict]:
    """Aggregate forward test results by week and strategy.

    Rolling averages are computed for each strategy over ``rolling_n`` weeks.
    Status colours are determined by comparing the rolling hit rate and
    drawdown against the provided thresholds.  ``green`` indicates the cohort is
    above both thresholds, ``red`` represents degradation, and ``yellow`` is a
    warning band within ten percentage points of the thresholds.
    """

    df = pd.DataFrame(list(rows))
    if df.empty:
        return []

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["week"] = df["created_at"].dt.to_period("W").dt.start_time
    grouped = (
        df.groupby(["week", "rule"])[["hit_forward", "roi_forward", "dd_forward"]]
        .mean()
        .reset_index()
        .sort_values(["rule", "week"])
    )

    grouped["hit_pct"] = grouped.groupby("rule")[["hit_forward"]].transform(
        lambda s: s.rolling(rolling_n, min_periods=1).mean()
    )
    grouped["roi"] = grouped.groupby("rule")[["roi_forward"]].transform(
        lambda s: s.rolling(rolling_n, min_periods=1).mean()
    )
    grouped["dd"] = grouped.groupby("rule")[["dd_forward"]].transform(
        lambda s: s.rolling(rolling_n, min_periods=1).mean()
    )

    out: List[Dict] = []
    for _, row in grouped.iterrows():
        hit = float(row.get("hit_pct")) if pd.notna(row.get("hit_pct")) else None
        roi = float(row.get("roi")) if pd.notna(row.get("roi")) else None
        dd = float(row.get("dd")) if pd.notna(row.get("dd")) else None
        status = "gray"
        if hit is not None and dd is not None:
            if hit < hit_red or dd > dd_red:
                status = "red"
            elif hit < hit_red + 10 or dd > dd_red - 10:
                status = "yellow"
            else:
                status = "green"
        out.append(
            CohortRow(
                week=pd.Timestamp(row["week"]),
                rule=str(row["rule"]),
                count=0,
                hit_pct=hit,
                roi=roi,
                dd=dd,
                status=status,
            ).__dict__,
        )
    return out
