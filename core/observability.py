"""
observability.py
----------------
Structured logging + metrics for the RAG pipeline.

Design goals
~~~~~~~~~~~~
* Zero external dependencies — uses only stdlib `logging` + `time`.
* Structured JSON logs (one object per event) so they can be ingested
  by any log aggregator (Loki, CloudWatch, Datadog, Elastic, …).
* A lightweight in-process metrics store with Prometheus-compatible
  output at  GET /metrics  (plain text).
* Context managers + decorators for timing any code block.

Exported helpers
~~~~~~~~~~~~~~~~
    get_logger(name)          → stdlib Logger that emits JSON
    log_event(event, **kw)    → emit a structured log line to the root logger
    timed(label)              → context manager; logs duration on exit
    MetricsStore              → singleton; counters + histograms
    metrics                   → the singleton instance
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record on a single line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields added via logger.info("…", extra={…})
        for key, val in record.__dict__.items():
            if key not in {
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "taskName",
                "thread", "threadName",
            }:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "rag") -> logging.Logger:
    """Return a logger that writes JSON to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


_root_logger = get_logger("rag")


def log_event(event: str, level: str = "info", **kwargs: Any) -> None:
    """Emit a named event with arbitrary key-value pairs."""
    fn = getattr(_root_logger, level.lower(), _root_logger.info)
    fn(event, extra=kwargs)


# ---------------------------------------------------------------------------
# In-process metrics store
# ---------------------------------------------------------------------------

class MetricsStore:
    """
    Thread-safe counters + duration histograms.

    Buckets for durations (in seconds):
        0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, +Inf
    """

    _BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    # --- counters ---

    def inc(self, name: str, amount: int = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += amount

    def counter(self, name: str, **labels: str) -> int:
        return self._counters[self._key(name, labels)]

    # --- histograms ---

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def histogram_summary(self, name: str, **labels: str) -> Dict[str, Any]:
        key = self._key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0}
        s = sorted(values)
        n = len(s)
        return {
            "count": n,
            "sum":   round(sum(s), 4),
            "p50":   round(s[n // 2], 4),
            "p95":   round(s[int(n * 0.95)], 4),
            "p99":   round(s[int(n * 0.99)], 4),
            "max":   round(s[-1], 4),
        }

    # --- Prometheus text export ---

    def prometheus_text(self) -> str:
        lines: List[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {val}")
            for key, values in sorted(self._histograms.items()):
                base = key.split("{")[0]
                labels_str = key[len(base):]  # e.g. '{method="query"}'
                ls = labels_str.rstrip("}") + ',' if labels_str else "{"
                lines.append(f"# TYPE {base} histogram")
                s = sorted(values)
                cumulative = 0
                for bucket in self._BUCKETS:
                    count = sum(1 for v in s if v <= bucket)
                    cumulative = count
                    b = "+Inf" if bucket == float("inf") else str(bucket)
                    lines.append(f'{base}_bucket{{{ls}le="{b}"}} {count}')
                lines.append(f"{base}_count{labels_str} {len(values)}")
                lines.append(f"{base}_sum{labels_str} {sum(values):.4f}")
        return "\n".join(lines)

    # --- helpers ---

    @staticmethod
    def _key(name: str, labels: Dict[str, str]) -> str:
        if not labels:
            return name
        pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{pairs}}}"


metrics = MetricsStore()


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str, log: bool = True, **extra: Any) -> Generator[dict, None, None]:
    """
    Context manager that measures wall-clock duration.

    Usage::

        with timed("retrieval", query=q) as t:
            docs = retriever.retrieve(q)
        # t["duration_ms"] is now set

    Also records to the histogram ``rag_duration_seconds{operation=label}``.
    """
    ctx: dict = {}
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        elapsed = time.perf_counter() - start
        ctx["duration_ms"] = round(elapsed * 1000, 2)
        metrics.observe("rag_duration_seconds", elapsed, operation=label)
        if log:
            log_event(label, duration_ms=ctx["duration_ms"], **extra)
