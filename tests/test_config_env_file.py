import importlib
import sys


def test_config_loads_export_env(tmp_path, monkeypatch):
    env_file = tmp_path / "petra.env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line should be ignored",
                "export DATA_PROVIDER=schwab",
                "export SCHWAB_CLIENT_ID=abc123",
                "SCHWAB_CLIENT_SECRET=shh # inline comment",
                'export SCHWAB_REDIRECT_URI="https://example.com/callback"',
                "export SCHWAB_ACCOUNT_ID=987654321",
            ]
        )
    )

    for key in [
        "DATA_PROVIDER",
        "SCHWAB_CLIENT_ID",
        "SCHWAB_CLIENT_SECRET",
        "SCHWAB_REDIRECT_URI",
        "SCHWAB_ACCOUNT_ID",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("PETRA_ENV_FILE", str(env_file))

    sys.modules.pop("config", None)
    config = importlib.import_module("config")

    try:
        settings = config.settings
    finally:
        sys.modules.pop("config", None)

    assert settings.data_provider.lower() == "schwab"
    assert settings.schwab_client_id == "abc123"
    assert settings.schwab_client_secret == "shh"
    assert settings.schwab_redirect_uri == "https://example.com/callback"
    assert settings.schwab_account_id == "987654321"
