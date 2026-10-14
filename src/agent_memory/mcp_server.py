"""MCP server: gives any MCP client (Claude Code, Claude Desktop, other agents) long-term memory.

Run it over stdio:
    memory-mcp                                     # database at ~/.agent-memory.db
    AGENT_MEMORY_DB=/path/to/team.db memory-mcp

Claude Code: claude mcp add memory -- memory-mcp
`remember` makes two small LLM calls (extract + decide); `recall` makes none.
"""

from __future__ import annotations

import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from agent_memory.engine import MemoryEngine
from agent_memory.llm import LLM
from agent_memory.store import MemoryStore

server = MCPServer(
    name="agent-memory",
    instructions="Long-term memory about the user. Call recall before answering questions about the user's "
                 "life or preferences, and remember when the user shares something worth keeping.",
)
_engine: MemoryEngine | None = None
_lock = threading.Lock()  # tools run in worker threads; one SQLite connection, one writer at a time


def engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        db = os.environ.get("AGENT_MEMORY_DB", str(Path.home() / ".agent-memory.db"))
        # Memory writes are frequent background work, so a small fast model is the default.
        _engine = MemoryEngine(store=MemoryStore(db), llm=LLM(os.environ.get("AGENT_MEMORY_MODEL", "claude-haiku-5-5")))
    return _engine


def set_engine(e: MemoryEngine) -> None:
    """For tests and embedding: use a given engine instead of the on-disk default."""
    global _engine
    _engine = e


@server.tool()
def remember(text: str, user_id: str = "default") -> dict[str, Any]:
    """Store what the user said about themselves. Facts that replace older ones (a new city, a new job) are
    recorded as updates and the old fact is kept as history."""
    with _lock:
        r = engine().remember(user_id, text, date.today().isoformat())
    return {"added": r.added, "updated": [{"old": o, "new": n} for o, n in r.updated], "skipped": r.skipped}


@server.tool()
def recall(query: str, user_id: str = "default", k: int = 8) -> list[dict[str, Any]]:
    """Find memories relevant to a query. Each result says whether it is current or no longer true."""
    with _lock:
        hits = engine().recall(user_id, query, k)
    return [{**h.memory.to_dict(), "score": round(h.score, 3), "found_via": h.via} for h in hits]


@server.tool()
def list_memories(user_id: str = "default", include_history: bool = False) -> list[dict[str, Any]]:
    """Everything remembered about a user (current facts only unless include_history is true)."""
    with _lock:
        return [m.to_dict() for m in engine().store.all(user_id, include_superseded=include_history)]


@server.tool()
def list_entities(user_id: str = "default") -> list[dict[str, Any]]:
    """People, pets, places and organizations in the user's memory, with how many memories mention each."""
    with _lock:
        return engine().store.entities(user_id)


@server.tool()
def forget(memory_id: str) -> dict[str, Any]:
    """Permanently delete one memory (for example when the user asks to be forgotten)."""
    with _lock:
        return {"deleted": engine().store.forget(memory_id), "memory_id": memory_id}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
