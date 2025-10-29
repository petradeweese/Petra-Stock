import math

from services.forecast_matcher import SimilarityScorer


def test_similarity_scorer_normalizes_weights(monkeypatch):
    monkeypatch.setenv("FORECAST_W5", "0.6")
    monkeypatch.setenv("FORECAST_W30", "0.3")
    monkeypatch.setenv("FORECAST_W1D", "0.1")

    scorer = SimilarityScorer()
    breakdown, final = scorer.score({"5m": 0.8, "30m": 0.6, "1d": 0.2})

    weights = breakdown["weights"]
    assert math.isclose(weights["5m"], 0.6, rel_tol=1e-6)
    assert math.isclose(weights["30m"], 0.3, rel_tol=1e-6)
    assert math.isclose(weights["1d"], 0.1, rel_tol=1e-6)
    expected = 0.6 * 0.8 + 0.3 * 0.6 + 0.1 * 0.2
    assert math.isclose(final, expected, rel_tol=1e-6)
    assert math.isclose(final, breakdown["final_score"], rel_tol=1e-6)


def test_similarity_scorer_renormalizes_missing_frames(monkeypatch):
    for key in ("FORECAST_W5", "FORECAST_W30", "FORECAST_W1D"):
        monkeypatch.delenv(key, raising=False)

    scorer = SimilarityScorer()
    breakdown, final = scorer.score({"30m": 0.75})

    weights = breakdown["weights"]
    assert math.isclose(weights["30m"], 1.0, rel_tol=1e-6)
    assert math.isclose(final, 0.75, rel_tol=1e-6)
    assert math.isclose(breakdown["S30m"], 0.75, rel_tol=1e-6)
