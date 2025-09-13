import pandas as pd

from cohorts import cohort_rollup


def test_cohort_rollup_status_and_rolling():
    rows = [
        {
            "created_at": "2024-01-01T12:00:00",
            "rule": "r1",
            "hit_forward": 100.0,
            "roi_forward": 2.0,
            "dd_forward": 10.0,
        },
        {
            "created_at": "2024-01-03T09:30:00",
            "rule": "r1",
            "hit_forward": 100.0,
            "roi_forward": 2.0,
            "dd_forward": 10.0,
        },
        {
            "created_at": "2024-01-10T09:30:00",
            "rule": "r1",
            "hit_forward": 0.0,
            "roi_forward": -2.0,
            "dd_forward": 70.0,
        },
    ]
    result = cohort_rollup(rows)
    assert len(result) == 2
    first, second = result
    assert pd.Timestamp(first["week"]) == pd.Timestamp("2024-01-01")
    assert pd.Timestamp(second["week"]) == pd.Timestamp("2024-01-08")
    assert first["hit_pct"] == 100.0
    assert second["hit_pct"] == 50.0
    assert first["status"] == "green"
    assert second["status"] == "red"
