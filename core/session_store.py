"""
session_store.py
-----------------
Lightweight in-process store for short-term conversation history.

The agentic layer needs *some* notion of "what did the user just ask" to
do context-dependency detection and query rewriting (coreference
resolution for follow-ups like "what about that one?"). This is NOT a
general chat-memory system — it only keeps the last SESSION_HISTORY_TURNS
(query, answer) pairs per session_id, in memory, with a TTL.

Deliberately NOT persisted to disk: sessions are cheap to lose (worst
case, a follow-up gets treated as standalone), and keeping this simple
avoids adding a new datastore for what is a small, bounded cache.

Usage
-----
    from core.session_store import get_history, append_turn

    history = get_history(session_id)          # List[Turn], oldest first
    append_turn(session_id, query, answer)
"""

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, List, Optional

from config import SESSION_HISTORY_TURNS, SESSION_TTL_SECONDS


@dataclass
class Turn:
    query: str
    answer: str
    ts: float


class _SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, Deque[Turn]] = {}
        self._touched: Dict[str, float] = {}

    def get_history(self, session_id: Optional[str]) -> List[Turn]:
        if not session_id:
            return []
        with self._lock:
            self._evict_expired()
            turns = self._sessions.get(session_id)
            return list(turns) if turns else []

    def append_turn(self, session_id: Optional[str], query: str, answer: str) -> None:
        if not session_id:
            return
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=SESSION_HISTORY_TURNS)
            self._sessions[session_id].append(Turn(query=query, answer=answer, ts=time.time()))
            self._touched[session_id] = time.time()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._touched.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, last in self._touched.items()
            if now - last > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._touched.pop(sid, None)


_store = _SessionStore()


def get_history(session_id: Optional[str]) -> List[Turn]:
    return _store.get_history(session_id)


def append_turn(session_id: Optional[str], query: str, answer: str) -> None:
    _store.append_turn(session_id, query, answer)


def clear_session(session_id: str) -> None:
    _store.clear(session_id)
