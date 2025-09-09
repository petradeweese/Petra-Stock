from routes import _format_rule_summary


def test_format_rule_summary_includes_key_params():
    params = {
        'target_pct': 1.0,
        'stop_pct': 0.5,
        'max_tt_bars': 5,
        'scan_min_hit': 60,
        'vega_scale': 0.03,
    }
    summary = _format_rule_summary(params)
    assert 'MaxBars 5' in summary
    assert 'MinHit% 60%' in summary
    assert 'Vega 0.03' in summary

