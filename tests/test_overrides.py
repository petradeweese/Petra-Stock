from services.favorites_alerts import load_profile


def test_override_merges_over_global():
    global_json = '{"mode":"buying","direction_profiles":{"UP":{"delta_min":0.3,"delta_max":0.7,"target_delta":0.5}}}'
    override_json = '{"direction_profiles":{"UP":{"delta_max":0.6}}}'
    merged = load_profile(global_json, override_json)
    up = merged["direction_profiles"]["UP"]
    assert up["delta_min"] == 0.3
    assert up["delta_max"] == 0.6  # override applied
