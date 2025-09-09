class Config:
    def __init__(self, ini_path: str):
        self.ini_path = ini_path
        self._opts = {"script_location": "alembic", "sqlalchemy.url": ""}

    def get_main_option(self, key: str) -> str:
        return self._opts.get(key, "")

    def set_main_option(self, key: str, value: str) -> None:
        self._opts[key] = value
