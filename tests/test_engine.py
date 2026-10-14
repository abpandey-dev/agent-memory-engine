from agent_memory.engine import ANSWER_SCHEMA, DECIDE_SYSTEM, EXTRACT_SYSTEM, MemoryEngine, format_hits
from agent_memory.llm import LLM, LLMError

from fakes import FakeClient, mem


def engine_with(script):
    client = FakeClient(script)
    return MemoryEngine(llm=LLM("claude-haiku-5-5", client=client)), client


def test_first_conversation_adds_without_a_decision_call():
    e, client = engine_with(lambda kw: {"memories": [mem("The user lives in Pune.", ["city"], [("Pune", "place")])]})
    r = e.ingest("u", [{"role": "user", "content": "I'm based in Pune"}], "2026-06-01", "s1")
    assert r.added == ["m-0001"] and len(client.calls) == 1
    assert client.calls[0]["system"] == EXTRACT_SYSTEM


def test_update_supersedes_and_skip_does_nothing():
    state = {"n": 0}

    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            state["n"] += 1
            if state["n"] == 1:
                return {"memories": [mem("The user lives in Pune.", ["city", "home"])]}
            return {"memories": [mem("The user moved to Berlin.", ["city", "home"]),
                                 mem("The user lives in Pune.", ["city", "home"])]}
        assert kw["system"] == DECIDE_SYSTEM and "[m-0001] The user lives in Pune." in kw["messages"][0]["content"]
        return {"decisions": [{"candidate": 0, "action": "UPDATE", "target_id": "m-0001"},
                              {"candidate": 1, "action": "SKIP", "target_id": ""}]}

    e, _ = engine_with(script)
    e.ingest("u", [{"role": "user", "content": "I'm in Pune"}], "2026-06-01", "s1")
    r = e.ingest("u", [{"role": "user", "content": "We moved to Berlin"}], "2026-07-06", "s6")
    assert r.updated == [("m-0001", "m-0002")] and len(r.skipped) == 1
    assert not e.store.get("m-0001").current and e.store.get("m-0002").current


def test_update_with_bad_target_falls_back_to_add():
    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            return {"memories": [mem("The user likes tea.", ["drink"])]}
        return {"decisions": [{"candidate": 0, "action": "UPDATE", "target_id": "m-9999"}]}

    e, _ = engine_with(script)
    e.ingest("u", [{"role": "user", "content": "tea"}], "2026-06-01", "s1")
    r = e.ingest("u", [{"role": "user", "content": "tea again"}], "2026-06-08", "s2")
    assert r.added == ["m-0002"] and r.updated == []


def test_answer_uses_recalled_memories_and_structured_output():
    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            return {"memories": [mem("The user's dog is named Biscuit.", ["dog", "pet"], [("Biscuit", "pet")])]}
        assert kw["output_config"]["format"]["schema"] == ANSWER_SCHEMA
        assert "Biscuit" in kw["messages"][0]["content"] and "Today is 2026-08-01" in kw["system"]
        return {"known": True, "answer": "Biscuit", "memory_ids": ["m-0001"]}

    e, client = engine_with(script)
    e.ingest("u", [{"role": "user", "content": "our dog Biscuit"}], "2026-06-01", "s1")
    a = e.answer("u", "What's my pet called?", "2026-08-01")
    assert a.known and a.answer == "Biscuit" and a.context_memories == ["m-0001"]
    assert "fallbacks" not in client.calls[-1]  # Haiku 5.5 has no server-side fallbacks


def test_fallbacks_sent_for_opus_and_refusal_raises():
    client = FakeClient(lambda kw: "refusal")
    llm = LLM("claude-opus-5-5", client=client)
    try:
        llm.json("s", "u", {"type": "object"})
        raise AssertionError("expected LLMError")
    except LLMError as e:
        assert "declined" in str(e)
    assert client.calls[0]["fallbacks"] == "default" and client.calls[0]["betas"]


def test_format_hits_marks_history():
    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            return {"memories": [mem("The user lives in Pune.", ["city"])]}
        return {"decisions": [{"candidate": 0, "action": "UPDATE", "target_id": "m-0001"}]}

    e, _ = engine_with(script)
    e.ingest("u", [{"role": "user", "content": "x"}], "2026-06-01", "s1")
    e.ingest("u", [{"role": "user", "content": "y"}], "2026-07-01", "s2")
    text = format_hits(e.recall("u", "city"))
    assert "[m-0002] (from 2026-07-01; current)" in text and "NO LONGER TRUE since 2026-07-01" in text


def test_known_people_are_passed_to_extraction():
    calls = []

    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            calls.append(kw["messages"][0]["content"])
            return {"memories": [mem("The user's husband is Diego.", ["spouse"], [("Diego", "person")])]}
        return {"decisions": [{"candidate": 0, "action": "SKIP", "target_id": ""}]}

    e, _ = engine_with(script)
    e.ingest("u", [{"role": "user", "content": "my husband Diego"}], "2026-06-01", "s1")
    e.ingest("u", [{"role": "user", "content": "Diego got a new job"}], "2026-06-29", "s5")
    assert "KNOWN PEOPLE AND PETS" not in calls[0]
    assert "- Diego (person): The user's husband is Diego." in calls[1]
