"""Hybrid recall: BM25 over LLM-enriched memories, plus one hop through the entity graph.

Why no vector database? At write time the extractor attaches keywords to every memory: synonyms and
categories ("Biscuit is the user's beagle" gets dog, pet, puppy). That lets plain BM25 match "what's my
dog called?" without embeddings. The entity graph handles the other hard case, facts linked only by a
name: "my wife Ananya" in one conversation and "Ananya started as a paediatrician" weeks later.
The benchmark measures each part separately (`graph=False`).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from agent_memory.store import Memory, MemoryStore

STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was were be been being am i me my
mine we our ours you your he she it its they them their this that these those do does did have has had what which who
whom whose when where why how can could should would will just so than then there here about into over after before
again any all also not no yes very user""".split())  # "user" is in nearly every memory


def stem(w: str) -> str:
    """A tiny suffix stripper, enough to match lives/living/lived and travel/travelled/travelling."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"  # countries -> country
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            w = w[: -len(suffix)]
            if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
                w = w[:-1]  # travell -> travel, runn -> run
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w  # move/moved -> mov, live/living -> liv


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / len(docs) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


@dataclass
class Hit:
    memory: Memory
    score: float
    via: str  # "keywords" or "graph:<entity>"


def memory_doc(m: Memory) -> list[str]:
    return tokenize(" ".join([m.text, " ".join(m.keywords), " ".join(m.entities)]))


class Retriever:
    def __init__(self, store: MemoryStore, graph: bool = True, include_superseded: bool = True,
                 superseded_weight: float = 0.6, graph_weight: float = 0.5):
        self.store, self.graph = store, graph
        self.include_superseded = include_superseded
        self.superseded_weight, self.graph_weight = superseded_weight, graph_weight

    def search(self, user_id: str, query: str, k: int = 8) -> list[Hit]:
        memories = self.store.all(user_id, include_superseded=self.include_superseded)
        if not memories:
            return []
        by_id = {m.id: m for m in memories}
        raw = BM25([memory_doc(m) for m in memories]).scores(tokenize(query))
        top = max(raw) or 1.0
        hits: dict[str, Hit] = {}
        for m, s in zip(memories, raw):
            if s <= 0:
                continue
            score = s / top + 0.02 * m.importance
            if not m.current:
                score *= self.superseded_weight  # history stays findable, but ranks below the current fact
            hits[m.id] = Hit(m, score, "keywords")

        if self.graph:
            # One hop: memories that share a named entity with a strong keyword hit.
            seeds = sorted(hits.values(), key=lambda h: -h.score)[:3]
            for seed in seeds:
                for ent in seed.memory.entities:
                    if ent.lower() == "user":
                        continue
                    for mid in self.store.entity_memories(user_id, ent):
                        if mid not in by_id:
                            continue
                        m = by_id[mid]
                        score = seed.score * self.graph_weight * (1 if m.current else self.superseded_weight)
                        if mid not in hits or hits[mid].score < score:
                            hits[mid] = Hit(m, score, f"graph:{ent}")

        ranked = sorted(hits.values(), key=lambda h: (-h.score, h.memory.id))
        return ranked[:k]
