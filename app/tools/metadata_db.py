"""
Disk-backed metadata store using SQLite.
Replaces in-memory list[ChunkMetadata] to avoid OOM on low-RAM servers.
Each row is keyed by its FAISS vector index (0-based rowid).
"""

import json
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

DB_FILENAME = "metadata.db"


def json_to_sqlite(json_path: str, db_path: str) -> int:
    """Convert metadata.json → metadata.db (SQLite). Returns row count."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE chunks (
            idx INTEGER PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)

    with open(json_path, "r") as f:
        raw = json.load(f)

    conn.executemany(
        "INSERT INTO chunks (idx, data) VALUES (?, ?)",
        [(i, json.dumps(m)) for i, m in enumerate(raw)],
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    logger.info(f"Converted {count} metadata entries to SQLite at {db_path}")
    return count


class MetadataDB:
    """Read-only SQLite-backed metadata lookup by FAISS index."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        return self._conn

    def get(self, idx: int) -> Optional[dict]:
        """Get metadata dict by FAISS vector index."""
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT data FROM chunks WHERE idx = ?", (idx,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_batch(self, indices: list[int]) -> dict[int, dict]:
        """Get multiple metadata dicts by indices."""
        conn = self._ensure_conn()
        placeholders = ",".join("?" * len(indices))
        rows = conn.execute(
            f"SELECT idx, data FROM chunks WHERE idx IN ({placeholders})",
            indices,
        ).fetchall()
        return {idx: json.loads(data) for idx, data in rows}

    def count(self) -> int:
        conn = self._ensure_conn()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_by_country(self, country: str) -> int:
        """Count chunks for a specific country."""
        conn = self._ensure_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE UPPER(json_extract(data, '$.country')) = ?",
            (country.upper(),),
        ).fetchone()[0]

    def get_all_countries(self) -> list[str]:
        """Get distinct country codes."""
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT DISTINCT json_extract(data, '$.country') FROM chunks"
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def iter_by_country(
        self, country: str, active_only: bool = True
    ) -> list[tuple[int, dict]]:
        """Iterate all chunks for a country. Returns (idx, parsed_dict) pairs."""
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT idx, data FROM chunks WHERE UPPER(json_extract(data, '$.country')) = ?",
            (country.upper(),),
        ).fetchall()
        results = []
        for idx, raw in rows:
            d = json.loads(raw)
            if active_only and not d.get("is_active", True):
                continue
            results.append((idx, d))
        return results

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
