"""SQLite-backed thread checkpoints and owner-scoped answer preferences."""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from threading import RLock
from typing import TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore
from langsmith import tracing_context


EMBEDDING_DIM = 128
HISTORY_LIMIT = 8
PREFERENCE_TTL_MINUTES = 30 * 24 * 60
PREFERENCES = {
    "concise": "简洁回答 关键要点 concise brief key points",
    "detailed": "详细回答 解释说明 detailed explanation",
    "steps": "按步骤回答 办理流程 step by step procedure",
}
_TEXT = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")


def embed(texts: list[str]) -> list[list[float]]:
    """Build reproducible, normalized text vectors without a model download."""
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIM
        for token in _TEXT.findall(text.lower()):
            if token.isascii():
                features = [token]
            else:
                features = list(token)
                features.extend(token[i:i + 2] for i in range(len(token) - 1))
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=4).digest()
                vector[int.from_bytes(digest, "big") % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        else:
            vector[0] = 1.0
        vectors.append(vector)
    return vectors


class HistoryState(TypedDict):
    turns: list[dict[str, str]]


class _CheckpointConnection(sqlite3.Connection):
    """Commit the native delete and state update in one SQLite transaction."""

    _replacing = False

    def commit(self):
        if not self._replacing:
            super().commit()

    @contextmanager
    def replace_snapshot(self):
        self.execute("BEGIN IMMEDIATE")
        self._replacing = True
        try:
            yield
            super().commit()
        except BaseException:
            self.rollback()
            raise
        finally:
            self._replacing = False


def _thread(session: str) -> dict:
    return {"configurable": {"thread_id": session}}


def _namespace(owner: str) -> tuple[str, str]:
    digest = hashlib.sha256(owner.encode()).hexdigest()
    return ("answer_preferences", digest)


class Memory:
    def __init__(self, connection: _CheckpointConnection, checkpoints: SqliteSaver, store: SqliteStore):
        self.connection = connection
        self.checkpoints = checkpoints
        self.store = store
        self._lock = RLock()
        builder = StateGraph(HistoryState)
        builder.add_node("save_turn", lambda state: {})
        builder.add_edge(START, "save_turn")
        builder.add_edge("save_turn", END)
        self.graph = builder.compile(checkpointer=checkpoints)

    def history(self, session: str) -> list[dict[str, str]]:
        with self._lock, tracing_context(enabled=False):
            return self.graph.get_state(_thread(session)).values.get("turns", [])

    def save_turn(self, session: str, question: str, answer: str) -> None:
        with self._lock, self.connection.replace_snapshot(), tracing_context(enabled=False):
            turns = [*self.history(session), {"question": question, "answer": answer}][-HISTORY_LIMIT:]
            self.checkpoints.delete_thread(session)
            self.graph.update_state(_thread(session), {"turns": turns}, as_node="save_turn")

    def remember(self, owner: str, preference: str) -> None:
        if preference not in PREFERENCES:
            raise ValueError("preference must be concise, detailed, or steps")
        self.store.put(
            _namespace(owner), "answer_style",
            {"kind": "preference", "field": "answer_style", "value": preference, "text": PREFERENCES[preference]},
            index=["text"],
            ttl=PREFERENCE_TTL_MINUTES,
        )

    def recall(self, owner: str, question: str) -> list[dict[str, str]]:
        self.store.sweep_ttl()
        items = self.store.search(_namespace(owner), query=question, limit=1)
        return [{key: item.value[key] for key in ("kind", "field", "value")} for item in items]


@contextmanager
def open_memory(path: Path) -> Iterator[Memory]:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {"dims": EMBEDDING_DIM, "embed": embed, "fields": ["text"]}
    with closing(sqlite3.connect(path, timeout=15, check_same_thread=False, factory=_CheckpointConnection)) as db:
        db.execute("PRAGMA secure_delete=ON")
        checkpoints = SqliteSaver(db, serde=JsonPlusSerializer(allowed_msgpack_modules=[]))
        checkpoints.setup()
        with SqliteStore.from_conn_string(str(path), index=index, ttl={"refresh_on_read": False}) as store:
            store.conn.execute("PRAGMA foreign_keys=ON")
            store.setup()
            yield Memory(db, checkpoints, store)
