from agent_memory.retrieval import Retriever, tokenize
from agent_memory.store import MemoryStore


def seeded():
    s = MemoryStore()
    s.add("u", "The user lives in Pune.", "fact", ["city", "home", "location"], 4, "2026-06-01", "s1", [{"name": "Pune", "type": "place"}])
    s.add("u", "The user's husband is Diego.", "fact", ["spouse", "partner", "married", "wife"], 5, "2026-06-08", "s2",
          [{"name": "Diego", "type": "person"}])
    s.add("u", "The user's beagle is named Biscuit.", "fact", ["dog", "pet", "puppy", "animal"], 4, "2026-06-15", "s3",
          [{"name": "Biscuit", "type": "pet"}])
    s.add("u", "Diego started a new job as a veterinarian.", "event", ["work", "career", "occupation"], 3, "2026-06-29",
          "s5", [{"name": "Diego", "type": "person"}])
    s.add("u", "The user moved to Berlin.", "event", ["city", "home", "relocation", "location"], 4, "2026-07-06", "s6",
          [{"name": "Berlin", "type": "place"}])
    s.supersede("m-0001", "m-0005", "2026-07-06")
    s.add("other", "The user lives in Rome.", "fact", ["city"], 4, "2026-06-01", "x", [])
    return s


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("What countries has the user visited?") == ["country", "visit"]
    assert tokenize("living lives lived") == ["liv"] * 3
    assert tokenize("move moved travel travelled travelling") == ["mov", "mov", "travel", "travel", "travel"]
    assert tokenize("dogs class") == ["dog", "class"]


def test_supersede_keeps_history():
    s = seeded()
    old = s.get("m-0001")
    assert not old.current and old.superseded_by == "m-0005" and old.valid_until == "2026-07-06"
    assert "no longer true since 2026-07-06" in old.to_dict()["status"]
    assert [m.id for m in s.all("u", include_superseded=False)] == ["m-0002", "m-0003", "m-0004", "m-0005"]


def test_keywords_let_paraphrased_questions_match():
    hits = Retriever(seeded()).search("u", "What's my dog called?")
    assert hits[0].memory.id == "m-0003"


def test_current_fact_outranks_history_but_history_is_still_found():
    ids = [h.memory.id for h in Retriever(seeded()).search("u", "Which city is my home?")]
    assert ids.index("m-0005") < ids.index("m-0001")


def test_graph_hop_links_facts_that_share_only_a_name():
    q = "What does my husband do for work?"
    with_graph = Retriever(seeded()).search("u", q, k=3)
    assert "m-0004" in [h.memory.id for h in with_graph]
    assert any(h.via == "graph:Diego" or h.memory.id == "m-0002" for h in with_graph)


def test_users_are_isolated_and_forget_deletes():
    s = seeded()
    assert all(h.memory.user_id == "u" for h in Retriever(s).search("u", "city"))
    assert s.forget("m-0003") and s.get("m-0003") is None and not s.forget("m-0003")
    assert "Biscuit" not in [e["name"] for e in s.entities("u") if e["memories"]]
