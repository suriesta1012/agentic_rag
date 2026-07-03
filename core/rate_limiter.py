"""
rate_limiter.py
----------------
In-process, per-client token-bucket rate limiter.

Design goals (matching the rest of core/ — see observability.py):
  * Zero external dependencies (no Redis) — fine for a single-process local
    deployment; swap for a shared store if this ever runs behind multiple
    workers/replicas.
  * Thread/async-safe via a single Lock, same pattern as MetricsStore.
  * Token bucket (not a fixed window) so a client isn't punished for a
    legitimate burst, but sustained abuse is still capped — this matters
    here specifically because /query and /upload are expensive (LLM
    generation, embedding, PDF parsing), so a naive fixed window either
    over-throttles bursty-but-legitimate use or under-throttles a client
    that times requests to land just inside each window edge.
  * Two independent buckets per client — "default" (cheap endpoints like
    /health, /classify) and "expensive" (/query, /upload, /eval*) — so a
    client hammering the expensive endpoints can't burn through budget
    that should be available for cheap ones for the same client, and
    vice versa.

Usage
-----
    from core.rate_limiter import limiter

    allowed, retry_after = limiter.check(client_key, tier="expensive")
    if not allowed:
        raise HTTPException(429, ...)  # with Retry-After: retry_after
"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Tuple

from config import (
    RATE_LIMIT_BURST_MULTIPLIER,
    RATE_LIMIT_DEFAULT_PER_MINUTE,
    RATE_LIMIT_EXPENSIVE_PER_MINUTE,
)

# How long an idle client's buckets are kept before being evicted, to keep
# memory bounded for a long-running process seeing many distinct IPs.
_IDLE_EVICTION_SECONDS = 3600


@dataclass
class _Bucket:
    tokens: float
    capacity: float
    refill_per_second: float
    last_refill: float

    def consume(self, now: float, amount: float = 1.0) -> Tuple[bool, float]:
        """
        Refill based on elapsed time, then try to consume `amount` tokens.

        Returns (allowed, retry_after_seconds). retry_after_seconds is 0
        when allowed=True.
        """
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.last_refill = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0

        deficit = amount - self.tokens
        retry_after = deficit / self.refill_per_second if self.refill_per_second > 0 else 60.0
        return False, round(retry_after, 1)


class RateLimiter:
    """Per-client, per-tier token buckets."""

    _TIERS = {
        "default":   RATE_LIMIT_DEFAULT_PER_MINUTE,
        "expensive": RATE_LIMIT_EXPENSIVE_PER_MINUTE,
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}
        self._last_seen: Dict[str, float] = {}

    def _get_bucket(self, client_key: str, tier: str, now: float) -> _Bucket:
        key = (client_key, tier)
        bucket = self._buckets.get(key)
        if bucket is None:
            per_minute = self._TIERS.get(tier, RATE_LIMIT_DEFAULT_PER_MINUTE)
            capacity = per_minute * RATE_LIMIT_BURST_MULTIPLIER
            bucket = _Bucket(
                tokens=capacity,  # start full — don't punish the first request
                capacity=capacity,
                refill_per_second=per_minute / 60.0,
                last_refill=now,
            )
            self._buckets[key] = bucket
        return bucket

    def check(self, client_key: str, tier: str = "default") -> Tuple[bool, float]:
        """
        Returns (allowed, retry_after_seconds).

        `client_key` should identify the caller (IP, or X-Forwarded-For
        when behind a proxy). `tier` selects which budget to draw from —
        use "expensive" for LLM/embedding/ingestion endpoints.
        """
        now = time.monotonic()
        with self._lock:
            self._last_seen[client_key] = now
            self._maybe_evict(now)
            bucket = self._get_bucket(client_key, tier, now)
            return bucket.consume(now)

    def _maybe_evict(self, now: float) -> None:
        """Drop buckets for clients idle past _IDLE_EVICTION_SECONDS."""
        stale_clients = [
            client for client, last in self._last_seen.items()
            if now - last > _IDLE_EVICTION_SECONDS
        ]
        for client in stale_clients:
            self._last_seen.pop(client, None)
            for tier in self._TIERS:
                self._buckets.pop((client, tier), None)


limiter = RateLimiter()
