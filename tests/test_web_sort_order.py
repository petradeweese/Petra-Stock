import pandas as pd

import routes


def test_sort_by_lb95_roi_support_on_list():
    rows = [
        {"ticker": "B", "hit_lb95": 0.35, "avg_roi": 0.01, "support": 25},
        {"ticker": "A", "avg_roi": 0.04, "support": 30},
        {"ticker": "C", "hit_lb95": 0.35, "avg_roi": 0.02, "support": 5},
    ]

    payload = [row.copy() for row in rows]
    result = routes._sort_by_lb95_roi_support(payload)

    assert result is payload
    assert [r["ticker"] for r in result] == ["C", "B", "A"]
    assert result[-1]["hit_lb95"] == 0.0


def test_sort_by_lb95_roi_support_on_dataframe():
    df = pd.DataFrame(
        [
            {"ticker": "X", "hit_lb95": 0.6, "avg_roi": 0.03, "support": 40},
            {"ticker": "Y", "avg_roi": 0.05, "support": 10},
            {"ticker": "Z", "hit_lb95": 0.6, "avg_roi": 0.04, "support": 20},
        ]
    )

    sorted_df = routes._sort_by_lb95_roi_support(df)

    assert list(sorted_df["ticker"]) == ["Z", "X", "Y"]
    # Original DataFrame should remain unchanged apart from untouched order
    assert list(df["ticker"]) == ["X", "Y", "Z"]
