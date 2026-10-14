"""The memory engine: write path (extract -> decide -> store) and read path (recall -> answer).

Write path, per conversation:
  1. extract  - Claude turns the conversation into standalone memories, each with keywords and entities.
  2. decide   - each candidate is compared with related existing memories, and Claude chooses
                ADD (new), UPDATE (replaces an outdated memory, which is kept as history) or SKIP (already known).
                One call per conversation covers all candidates.
  3. store    - SQLite rows, plus links to the entities they mention (the graph).

Read path: hybrid recall (see retrieval.py), then Claude answers using only the recalled memories,
citing memory ids, and says it doesn't know when they don't contain the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent_memory.llm import DEFAULT_MODEL, LLM
from agent_memory.retrieval import Hit, Retriever
from agent_memory.store import MemoryStore

KINDS = ["fact", "preference", "event", "plan"]
ENTITY_TYPES = ["person", "pet", "place", "organization", "thing"]

EXTRACT_SYSTEM = """You maintain long-term memory about a user for an AI assistant.
Read the conversation and extract what is worth remembering about the USER for future conversations:
facts about their life, people and pets in it, preferences, changes ("we moved", "I stopped eating meat"),
and plans with dates.

Rules:
- Each memory is one standalone sentence in the third person ("The user's dog is a beagle named Biscuit."),
  understandable with no other context. Resolve pronouns and relative dates using the conversation date.
- Only extract things about the user and their own life. Ignore requests for help, general knowledge, and
  facts about other people's pets or homes unless they are part of the user's life.
- keywords: 4-8 lowercase words someone might use when asking about this later, including synonyms and
  categories that are NOT in the sentence (for a beagle: dog, pet, puppy, animal; for a holiday in Peru:
  travel, trip, country, abroad).
- If KNOWN PEOPLE AND PETS are listed, use them to say who someone is ("the user's husband Diego", not
  "a friend Diego"). Never guess a relationship that isn't stated or known.
- entities: named people, pets, places and organizations in the memory (not the user).
- importance: 1 (trivia) to 5 (core identity, health, family).
- If there is nothing worth remembering, return an empty list."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"},
                                           "type": {"type": "string", "enum": ENTITY_TYPES}},
                            "required": ["name", "type"],
                            "additionalProperties": False,
                        },
                    },
                    "importance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                },
                "required": ["text", "kind", "keywords", "entities", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

DECIDE_SYSTEM = """You curate an AI assistant's long-term memory about a user.
For each NEW candidate memory you see the EXISTING memories most related to it. Choose:
- ADD: new information. Also ADD when it adds to an existing memory without contradicting it
  (visiting a second country does not replace the first).
- UPDATE: it replaces an existing memory that is no longer true (moved city, new job title, changed diet,
  no longer has a pet). Give that memory's id as target_id; the old memory is kept as history.
- SKIP: already known; an existing memory says the same thing.
target_id is "" for ADD and SKIP."""

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "action": {"type": "string", "enum": ["ADD", "UPDATE", "SKIP"]},
                    "target_id": {"type": "string"},
                },
                "required": ["candidate", "action", "target_id"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = """You are a personal assistant with long-term memory of earlier conversations with this user.
Today is {today}. Answer the user's question using ONLY the context below; it is all you know about them.
Memories marked "no longer true" describe the past: use them only for questions about the past.
If the context does not contain the answer, set known to false and say you don't know; never guess.
Keep the answer short. List the ids of the memories you used (if any)."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "known": {"type": "boolean"},
        "answer": {"type": "string"},
        "memory_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["known", "answer", "memory_ids"],
    "additionalProperties": False,
}


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[tuple[str, str]] = field(default_factory=list)  # (old id, new id)
    skipped: list[str] = field(default_factory=list)


@dataclass
class Answer:
    known: bool
    answer: str
    memory_ids: list[str]
    context_memories: list[str] = field(default_factory=list)


def format_transcript(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def format_hits(hits: list[Hit]) -> str:
    lines = []
    for h in hits:
        m = h.memory
        status = "current" if m.current else f"NO LONGER TRUE since {m.valid_until}, replaced by {m.superseded_by}"
        lines.append(f"[{m.id}] (from {m.observed_at}; {status}) {m.text}")
    return "\n".join(lines) or "(no memories found)"


def answer_from_context(llm: LLM, question: str, context: str, today: str) -> Answer:
    out = llm.json(ANSWER_SYSTEM.format(today=today), f"<context>\n{context}\n</context>\n\nQuestion: {question}",
                   ANSWER_SCHEMA, max_tokens=2000)
    return Answer(known=out["known"], answer=out["answer"], memory_ids=out["memory_ids"])


class MemoryEngine:
    def __init__(self, store: MemoryStore | None = None, llm: LLM | None = None, model: str = DEFAULT_MODEL,
                 graph: bool = True, k: int = 8):
        self.store = store or MemoryStore()
        self.llm = llm or LLM(model)
        self.retriever = Retriever(self.store, graph=graph)
        self.k = k

    # --- write path -------------------------------------------------------------------------------

    def known_entities(self, user_id: str, limit: int = 30) -> str:
        """Who the user has mentioned before, so a bare name in a new conversation can be resolved."""
        lines = []
        for e in self.store.entities(user_id):
            if e["type"] not in ("person", "pet") or not e["memories"]:
                continue
            first = self.store.get(min(self.store.entity_memories(user_id, e["name"])))
            lines.append(f"- {e['name']} ({e['type']}): {first.text}")
            if len(lines) == limit:
                break
        return "\n".join(lines)

    def extract(self, messages: list[dict], date: str, user_id: str | None = None) -> list[dict]:
        known = self.known_entities(user_id) if user_id else ""
        context = f"KNOWN PEOPLE AND PETS:\n{known}\n\n" if known else ""
        out = self.llm.json(EXTRACT_SYSTEM,
                            f"{context}Conversation date: {date}\n\n{format_transcript(messages)}", EXTRACT_SCHEMA)
        return out["memories"]

    def ingest(self, user_id: str, messages: list[dict], date: str, session_id: str) -> IngestReport:
        report = IngestReport()
        candidates = self.extract(messages, date, user_id)
        if not candidates:
            return report

        current = Retriever(self.store, graph=False, include_superseded=False)
        related = [current.search(user_id, c["text"] + " " + " ".join(c["keywords"]), k=4) for c in candidates]
        decisions = {i: ("ADD", "") for i in range(len(candidates))}
        if any(related):
            blocks = []
            for i, (c, hits) in enumerate(zip(candidates, related)):
                existing = "\n".join(f"  [{h.memory.id}] {h.memory.text}" for h in hits) or "  (none)"
                blocks.append(f"Candidate {i}: {c['text']}\nExisting related memories:\n{existing}")
            out = self.llm.json(DECIDE_SYSTEM, "\n\n".join(blocks), DECIDE_SCHEMA, max_tokens=4000)
            for d in out["decisions"]:
                if 0 <= d["candidate"] < len(candidates):
                    decisions[d["candidate"]] = (d["action"], d["target_id"])

        for i, c in enumerate(candidates):
            action, target = decisions[i]
            if action == "SKIP":
                report.skipped.append(c["text"])
                continue
            m = self.store.add(user_id, c["text"], c["kind"], [k.lower() for k in c["keywords"]], c["importance"],
                               date, session_id, c["entities"])
            old = self.store.get(target) if action == "UPDATE" and target else None
            if old and old.current and old.user_id == user_id and old.id != m.id:
                self.store.supersede(old.id, m.id, date)
                report.updated.append((old.id, m.id))
            else:
                report.added.append(m.id)
        return report

    def remember(self, user_id: str, text: str, date: str, source: str = "manual") -> IngestReport:
        """Store something the user said directly (used by the MCP server)."""
        return self.ingest(user_id, [{"role": "user", "content": text}], date, source)

    # --- read path --------------------------------------------------------------------------------

    def recall(self, user_id: str, query: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(user_id, query, k or self.k)

    def answer(self, user_id: str, question: str, today: str) -> Answer:
        hits = self.recall(user_id, question)
        a = answer_from_context(self.llm, question, format_hits(hits), today)
        a.context_memories = [h.memory.id for h in hits]
        return a

    def export(self, user_id: str) -> str:
        return json.dumps([m.to_dict() for m in self.store.all(user_id)], indent=1)
