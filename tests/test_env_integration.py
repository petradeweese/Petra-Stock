from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_app_uses_env_file_for_schwab_settings(tmp_path):
    token_path = tmp_path / "schwab_tokens.json"
    token_path.write_text(json.dumps({"refresh_token": "refresh-from-path"}))

    env_file = tmp_path / "petra.env"
    env_file.write_text(
        "\n".join(
            [
                "DATA_PROVIDER=schwab",
                "SCHWAB_CLIENT_ID=client-from-file",
                "SCHWAB_CLIENT_SECRET=secret#with-comment-char",
                "SCHWAB_REDIRECT_URI=https://example.com/callback",
                "SCHWAB_ACCOUNT_ID=00001234",
                "SCHWAB_REFRESH_TOKEN=refresh-from-file",
                f"SCHWAB_TOKENS_PATH={token_path}",
                "",
            ]
        )
    )

    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        filter(None, [str(repo_root), os.environ.get("PYTHONPATH", "")])
    )

    script = r"""
import json
import os
import sys

env_file = sys.argv[1]
for key in list(os.environ):
    if key.startswith("SCHWAB_"):
        os.environ.pop(key, None)

os.environ.setdefault("DATA_PROVIDER", "schwab")
os.environ["PETRA_ENV_FILE"] = env_file

import main  # noqa: F401 - triggers application import
from config import settings

payload = {
    "client_id": settings.schwab_client_id,
    "client_secret": settings.schwab_client_secret,
    "redirect_uri": settings.schwab_redirect_uri,
    "account_id": settings.schwab_account_id,
    "refresh_token": settings.schwab_refresh_token,
    "tokens_path": os.getenv("SCHWAB_TOKENS_PATH"),
}

print(json.dumps(payload))
"""

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "DATA_PROVIDER": "schwab",
        "PETRA_ENV_FILE": str(env_file),
    }

    result = subprocess.run(
        [sys.executable, "-c", script, str(env_file)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )

    output_line = result.stdout.strip().splitlines()[-1]
    data = json.loads(output_line)

    assert data == {
        "client_id": "client-from-file",
        "client_secret": "secret#with-comment-char",
        "redirect_uri": "https://example.com/callback",
        "account_id": "00001234",
        "refresh_token": "refresh-from-path",
        "tokens_path": str(token_path),
    }
