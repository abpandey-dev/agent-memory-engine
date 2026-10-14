"""Benchmark: the memory engine against three ways of answering without one.

  no_memory     the question alone (shows what a stateless assistant gets right by luck)
  full_context  every past conversation pasted into the prompt (the brute-force upper bound; cost grows
                with history)
  chunk_rag     BM25 over raw user messages, top 8 (typical "RAG over chat logs")
  memory        this engine: extracted memories with history + graph recall, top 8
  memory_no_graph  ablation: same memories, recall without the entity-graph hop

Every method uses the same answering prompt and model; only the context differs. Scoring is deterministic
string matching on the expected values (no LLM judge), and "I don't know" is scored on the `known` flag.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass

from agent_memory.dataset import Persona, Question
from agent_memory.engine import MemoryEngine, answer_from_context, format_transcript
from agent_memory.llm import LLM, LLMError, Usage
from agent_memory.retrieval import BM25, tokenize
from agent_memory.store import MemoryStore

METHODS = ["no_memory", "full_context", "chunk_rag", "memory", "memory_no_graph"]


def normalize(text: str) -> str:
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text.lower())
    return " ".join(re.sub(r"[^a-z0-9\- ]", " ", text).split())


def is_correct(q: Question, known: bool, answer: str) -> bool:
    if not q.known:
        return not known
    if not known:
        return False
    a = normalize(answer)
    return all(any(normalize(alias) in a for alias in group) for group in q.expected)


@dataclass
class Result:
    method: str
    question_id: str
    type: str
    correct: bool
    known: bool
    answer: str
    input_tokens: int
    output_tokens: int
    error: str | None = None


def full_context(p: Persona) -> str:
    return "\n\n".join(f"--- Conversation on {s.date} ---\n{format_transcript(s.messages)}" for s in p.sessions)


def chunk_context(p: Persona, question: str, k: int = 8) -> str:
    chunks = [(s.date, m["content"]) for s in p.sessions for m in s.messages if m["role"] == "user"]
    scores = BM25([tokenize(c) for _, c in chunks]).scores(tokenize(question))
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return "\n".join(f"(user said on {chunks[i][0]}) {chunks[i][1]}" for i in sorted(top) if scores[i] > 0) or \
        "(nothing found)"


class Runner:
    def __init__(self, model: str, client=None, verbose: bool = True):
        self.model = model
        self.client = client
        self.verbose = verbose
        self.ingest_usage = Usage()

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def build_memory(self, p: Persona) -> MemoryStore:
        store = MemoryStore()
        llm = LLM(self.model, client=self.client)
        engine = MemoryEngine(store=store, llm=llm)
        for s in p.sessions:
            r = engine.ingest(p.user_id, s.messages, s.date, s.id)
            self.log(f"  {s.id}: +{len(r.added)} added, {len(r.updated)} updated, {len(r.skipped)} skipped")
        self.ingest_usage.add(llm.usage)
        return store

    def run(self, personas: list[Persona], methods: list[str]) -> list[Result]:
        results = []
        for p in personas:
            self.log(f"{p.name}: {len(p.sessions)} sessions, {len(p.questions)} questions")
            store = self.build_memory(p) if any(m.startswith("memory") for m in methods) else None
            for method in methods:
                llm = LLM(self.model, client=self.client)
                engine = MemoryEngine(store=store, llm=llm, graph=method != "memory_no_graph") if store else None
                for q in p.questions:
                    before, before_out = llm.usage.input_tokens, llm.usage.output_tokens
                    try:
                        if method == "no_memory":
                            a = answer_from_context(llm, q.question, "(no previous conversations)", p.today)
                        elif method == "full_context":
                            a = answer_from_context(llm, q.question, full_context(p), p.today)
                        elif method == "chunk_rag":
                            a = answer_from_context(llm, q.question, chunk_context(p, q.question), p.today)
                        else:
                            a = engine.answer(p.user_id, q.question, p.today)
                        results.append(Result(method, q.id, q.type, is_correct(q, a.known, a.answer), a.known,
                                              a.answer, llm.usage.input_tokens - before,
                                              llm.usage.output_tokens - before_out))
                    except LLMError as e:
                        results.append(Result(method, q.id, q.type, False, False, "", 0, 0, str(e)))
                self.log(f"  {method}: {sum(r.correct for r in results if r.method == method)} correct so far")
        return results


def summarize(results: list[Result], methods: list[str], model: str, ingest_usage: Usage) -> str:
    types = list(dict.fromkeys(r.type for r in results))
    head = "| Method | Overall | " + " | ".join(types) + " | Input tokens / question | Cost / question |"
    lines = [head, "|---" * (len(types) + 4) + "|"]
    for m in methods:
        rs = [r for r in results if r.method == m]
        by = defaultdict(list)
        for r in rs:
            by[r.type].append(r.correct)
        cells = [f"{sum(v) / len(v):.0%}" for t in types for v in [by[t]]]
        avg_in = sum(r.input_tokens for r in rs) / len(rs)
        c = Usage(len(rs), sum(r.input_tokens for r in rs), sum(r.output_tokens for r in rs)).cost(model)
        per_q = f"${c / len(rs):.5f}" if c is not None else "n/a"
        lines.append(f"| {m} | **{sum(r.correct for r in rs) / len(rs):.0%}** | " + " | ".join(cells)
                     + f" | {avg_in:,.0f} | {per_q} |")
    cost = ingest_usage.cost(model)
    lines.append("")
    lines.append(f"Memory building: {ingest_usage.calls} calls, {ingest_usage.input_tokens:,} input + "
                 f"{ingest_usage.output_tokens:,} output tokens"
                 + (f" (${cost:.4f})" if cost is not None else "") + f", model {model}.")
    return "\n".join(lines) + "\n"


def results_json(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
