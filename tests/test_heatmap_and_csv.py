import json

import routes


def test_heatmap_enriches_sector_with_mapping():
    rows = [
        {"ticker": "AAPL", "support": 25, "hit_lb95": 0.6, "avg_roi": 0.02},
        {"ticker": "XOM", "support": 30, "hit_lb95": 0.55, "avg_roi": 0.015},
        {"ticker": "ZZZZ", "support": 40, "hit_lb95": 0.5, "avg_roi": 0.01},
    ]

    data = routes._build_heatmap(rows)
    sectors = data.get("index", [])

    assert "Information Technology" in sectors
    assert "Energy" in sectors
    assert len(set(sectors)) >= 2


def test_rows_to_csv_table_includes_new_metrics():
    row = {
        "ticker": "AAPL",
        "direction": "UP",
        "avg_roi_pct": 12.3,
        "hit_pct": 55.5,
        "hit_lb95": 0.45,
        "support": 30,
        "avg_tt": 4.0,
        "avg_dd_pct": 1.2,
        "stability": 9.5,
        "sharpe": 1.8,
        "rule": "demo",
        "stop_pct": 0.2,
        "timeout_pct": 0.05,
        "confidence": 78,
        "confidence_label": "High",
        "recent3": [{"date": "2024-01-01", "roi": 0.1, "tt": 3, "outcome": "hit"}],
    }

    headers, csv_rows = routes._rows_to_csv_table([row])
    assert headers == routes._CSV_EXPORT_COLUMNS
    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    for field in [
        "hit_lb95",
        "stop_pct",
        "timeout_pct",
        "confidence",
        "confidence_label",
    ]:
        idx = headers.index(field)
        assert csv_row[idx] == row[field]

    recent_idx = headers.index("recent3")
    assert json.loads(csv_row[recent_idx]) == row["recent3"]
