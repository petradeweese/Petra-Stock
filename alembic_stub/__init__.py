from __future__ import annotations

op = None

class _Op:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql: str) -> None:
        self.conn.execute(sql)


def set_connection(conn) -> None:
    global op
    op = _Op(conn)
