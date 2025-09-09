import importlib.util
import sqlite3
from pathlib import Path

from . import set_connection


def upgrade(cfg, revision: str) -> None:
    url = cfg.get_main_option("sqlalchemy.url")
    if not url.startswith("sqlite:///"):
        raise ValueError("Only sqlite URLs are supported in this stub")
    db_path = url.replace("sqlite:///", "", 1)
    conn = sqlite3.connect(db_path)
    try:
        set_connection(conn)
        versions_dir = Path(cfg.get_main_option("script_location")) / "versions"
        for path in sorted(versions_dir.glob("*.py")):
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "revision") or not hasattr(module, "down_revision"):
                raise AttributeError(
                    f"Migration {path} is missing revision or down_revision"
                )
            if hasattr(module, "upgrade"):
                module.upgrade()
        conn.commit()
    finally:
        conn.close()
