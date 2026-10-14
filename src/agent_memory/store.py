"""SQLite store for memories, the entities they mention, and their history.

Nothing is hard-deleted when facts change: an updated memory is marked superseded (with the date and the
memory that replaced it), so the engine can answer both "where do I live?" and "where did I live before?".
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    text          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    keywords      TEXT NOT NULL,          -- JSON list, written by the extractor (synonyms, categories)
    importance    INTEGER NOT NULL,
    observed_at   TEXT NOT NULL,          -- date of the conversation it came from
    source        TEXT NOT NULL,          -- session id
    superseded_by TEXT,
    valid_until   TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    name      TEXT NOT NULL,
    type      TEXT NOT NULL,
    UNIQUE (user_id, name_norm)
);
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL REFERENCES memories(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (memory_id, entity_id)
);
"""


@dataclass
class Memory:
    id: str
    user_id: str
    text: str
    kind: str
    keywords: list[str]
    importance: int
    observed_at: str
    source: str
    entities: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    valid_until: str | None = None

    @property
    def current(self) -> bool:
        return self.superseded_by is None

    def to_dict(self) -> dict:
        d = {"id": self.id, "text": self.text, "kind": self.kind, "observed_at": self.observed_at,
             "entities": self.entities, "status": "current"}
        if not self.current:
            d["status"] = f"no longer true since {self.valid_until} (replaced by {self.superseded_by})"
        return d


def norm(name: str) -> str:
    return " ".join(name.lower().split())


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        # The MCP server runs tools in worker threads; it serialises access with a lock (see mcp_server.py).
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def _next_id(self) -> str:
        (n,) = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        return f"m-{n + 1:04d}"

    def add(self, user_id: str, text: str, kind: str, keywords: list[str], importance: int,
            observed_at: str, source: str, entities: list[dict]) -> Memory:
        mid = self._next_id()
        self.db.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
                        (mid, user_id, text, kind, json.dumps(keywords), importance, observed_at, source))
        for e in entities:
            self.db.execute("INSERT OR IGNORE INTO entities (user_id, name_norm, name, type) VALUES (?,?,?,?)",
                            (user_id, norm(e["name"]), e["name"], e["type"]))
            (eid,) = self.db.execute("SELECT id FROM entities WHERE user_id=? AND name_norm=?",
                                     (user_id, norm(e["name"]))).fetchone()
            self.db.execute("INSERT OR IGNORE INTO memory_entities VALUES (?,?)", (mid, eid))
        self.db.commit()
        return self.get(mid)

    def supersede(self, old_id: str, new_id: str, when: str) -> None:
        self.db.execute("UPDATE memories SET superseded_by=?, valid_until=? WHERE id=? AND superseded_by IS NULL",
                        (new_id, when, old_id))
        self.db.commit()

    def forget(self, memory_id: str) -> bool:
        """Hard delete, for when a user asks for something to be forgotten (right to erasure)."""
        self.db.execute("DELETE FROM memory_entities WHERE memory_id=?", (memory_id,))
        cur = self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def _row(self, r) -> Memory:
        ents = [n for (n,) in self.db.execute(
            "SELECT e.name FROM entities e JOIN memory_entities me ON me.entity_id=e.id WHERE me.memory_id=? "
            "ORDER BY e.name", (r[0],))]
        return Memory(id=r[0], user_id=r[1], text=r[2], kind=r[3], keywords=json.loads(r[4]), importance=r[5],
                      observed_at=r[6], source=r[7], superseded_by=r[8], valid_until=r[9], entities=ents)

    def get(self, memory_id: str) -> Memory | None:
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row(r) if r else None

    def all(self, user_id: str, include_superseded: bool = True) -> list[Memory]:
        q = "SELECT * FROM memories WHERE user_id=?" + ("" if include_superseded else " AND superseded_by IS NULL")
        return [self._row(r) for r in self.db.execute(q + " ORDER BY id", (user_id,))]

    def entity_memories(self, user_id: str, entity_name: str) -> list[str]:
        return [m for (m,) in self.db.execute(
            "SELECT me.memory_id FROM memory_entities me JOIN entities e ON e.id=me.entity_id "
            "WHERE e.user_id=? AND e.name_norm=?", (user_id, norm(entity_name)))]

    def entities(self, user_id: str) -> list[dict]:
        return [{"name": n, "type": t, "memories": c} for n, t, c in self.db.execute(
            "SELECT e.name, e.type, COUNT(me.memory_id) FROM entities e LEFT JOIN memory_entities me "
            "ON me.entity_id=e.id WHERE e.user_id=? GROUP BY e.id ORDER BY COUNT(me.memory_id) DESC, e.name",
            (user_id,))]
