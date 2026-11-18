"""A SQLite file of {key: text} that runs share. The solver's reply cache
and the sandbox's result cache are each one."""

import sqlite3
import threading
from pathlib import Path


class SqliteCache:
    """{key: text} in one table of an sqlite file."""

    def __init__(self, path: str | Path, table: str, column: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # several processes write at once, wal lets readers carry on
        self._db = sqlite3.connect(path, timeout=300, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            f"create table if not exists {table} (key text primary key, "
            f"{column} text not null)"
        )
        self._db.commit()
        self._lock = threading.Lock()
        self._select = f"select {column} from {table} where key = ?"
        self._insert = f"insert or ignore into {table} values (?, ?)"

    def get(self, key: str) -> str | None:
        """the text stored under key, None if there's none."""
        with self._lock:
            row = self._db.execute(self._select, (key,)).fetchone()
        return row[0] if row else None

    def put(self, pairs: dict[str, str]):
        """stores {key: text} in one commit. a commit per key is slow
        enough to eat what the cache saves."""
        with self._lock:
            self._db.executemany(self._insert, list(pairs.items()))
            self._db.commit()
