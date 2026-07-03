"""
chunk_hash_store.py
-------------------
Persists a mapping of  chunk_id → SHA-256(content)  to a local SQLite
database so we can detect which chunks changed between document versions
and skip re-embedding unchanged ones.

Schema
------
  chunk_hashes(chunk_id TEXT PRIMARY KEY, content_hash TEXT, source TEXT,
               chunk_index INTEGER, updated_at REAL)

chunk_id  = "<source_filename>:<chunk_index>"  (stable across uploads of
             the same document so we can diff old vs new)
"""

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CHROMA_PERSIST_DIR

_DB_PATH = Path(CHROMA_PERSIST_DIR) / "chunk_hashes.db"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _conn():
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create the table if it doesn't exist yet."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS chunk_hashes (
                chunk_id    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                updated_at  REAL NOT NULL
            )
        """)


def _chunk_id(source: str, chunk_index: int) -> str:
    return f"{source}:{chunk_index}"


def get_stored_hashes(source: str) -> Dict[str, str]:
    """Return {chunk_id: content_hash} for every chunk of *source*."""
    with _conn() as con:
        rows = con.execute(
            "SELECT chunk_id, content_hash FROM chunk_hashes WHERE source = ?",
            (source,),
        ).fetchall()
    return {r["chunk_id"]: r["content_hash"] for r in rows}


def upsert_hashes(source: str, chunks_with_content: List[Tuple[int, str]]) -> None:
    """
    Save or update hashes for *source*.

    Args:
        source: filename / doc identifier.
        chunks_with_content: list of (chunk_index, page_content) tuples.
    """
    now = time.time()
    with _conn() as con:
        con.executemany(
            """
            INSERT INTO chunk_hashes (chunk_id, content_hash, source, chunk_index, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                updated_at   = excluded.updated_at
            """,
            [
                (_chunk_id(source, idx), _content_hash(content), source, idx, now)
                for idx, content in chunks_with_content
            ],
        )


def delete_hashes_for_source(source: str) -> int:
    """Remove all stored hashes for *source*. Returns rows deleted."""
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM chunk_hashes WHERE source = ?", (source,)
        )
    return cur.rowcount


def find_changed_chunks(
    source: str,
    new_chunks_with_content: List[Tuple[int, str]],
) -> Tuple[List[Tuple[int, str]], List[str]]:
    """
    Compare incoming chunks against stored hashes.

    Returns:
        changed_chunks  — list of (chunk_index, content) that are new or modified.
        stale_chunk_ids — chunk_ids that exist in DB but are absent from the
                          new document (i.e. chunks that were deleted/merged).
    """
    stored = get_stored_hashes(source)
    new_ids = {_chunk_id(source, idx) for idx, _ in new_chunks_with_content}

    changed: List[Tuple[int, str]] = []
    for idx, content in new_chunks_with_content:
        cid = _chunk_id(source, idx)
        if stored.get(cid) != _content_hash(content):
            changed.append((idx, content))

    stale_ids = [cid for cid in stored if cid not in new_ids]
    return changed, stale_ids


# Initialise on import
init_db()
