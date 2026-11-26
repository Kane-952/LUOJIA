"""Local, structured diagnostics for one Agent request."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger(__name__)
TRACE_ID = re.compile(r"[0-9a-f]{32}")
ACTIVE_TRACE: ContextVar[TurnTrace | None] = ContextVar("luojia_turn_trace", default=None)


class TraceStore:
    def __init__(self, root):
        self.path = Path(root) / "turn_traces.sqlite3"

    def write(self, payload):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=1)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL
                )""")
                db.execute("CREATE INDEX IF NOT EXISTS traces_created_at ON traces(created_at)")
                now = time.time()
                db.execute("DELETE FROM traces WHERE created_at < ?", (now - 7 * 86400,))
                db.execute("INSERT INTO traces VALUES (?, ?, ?)", (
                    payload["trace_id"], now, json.dumps(payload, ensure_ascii=False, allow_nan=False),
                ))
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            LOGGER.warning("Unable to persist turn trace %s (%s)", payload["trace_id"], type(exc).__name__)

    def read(self, trace_id):
        if not TRACE_ID.fullmatch(trace_id):
            raise ValueError("trace ID must contain 32 lowercase hexadecimal characters")
        if not self.path.is_file():
            return None
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return json.loads(row[0]) if row else None


class TurnTrace:
    def __init__(self, store, mode, transport, trace_id=None):
        self.store = store
        self.started = time.perf_counter()
        self.payload = {
            "schema_version": 1,
            "trace_id": trace_id or uuid.uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "transport": transport,
            "mode": mode,
            "events": [],
        }

    @contextmanager
    def activate(self):
        token = ACTIVE_TRACE.set(self)
        try:
            yield
        finally:
            ACTIVE_TRACE.reset(token)

    def event(self, stage, *, started=None, ended=None, **metadata):
        ended = time.perf_counter() if ended is None else ended
        started = ended if started is None else started
        self.payload["events"].append({
            "stage": stage,
            "offset_ms": round((started - self.started) * 1000, 3),
            "elapsed_ms": round((ended - started) * 1000, 3),
            **metadata,
        })

    def finish(self, **metadata):
        self.event("result", **metadata)
        self.payload.update(elapsed_ms=round((time.perf_counter() - self.started) * 1000, 3), **metadata)
        self.store.write(self.payload)


def trace_id():
    active = ACTIVE_TRACE.get()
    return active.payload["trace_id"] if active else None


def trace_event(stage, **metadata):
    active = ACTIVE_TRACE.get()
    if active:
        active.event(stage, **metadata)


def model_usage(usage):
    if not isinstance(usage, dict):
        return {}
    return {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens")
            if type(usage.get(key)) is int and usage[key] >= 0}
