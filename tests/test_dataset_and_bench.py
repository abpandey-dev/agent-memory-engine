from agent_memory.bench import Runner, chunk_context, is_correct, normalize, summarize
from agent_memory.dataset import dataset, make_persona
from agent_memory.engine import EXTRACT_SYSTEM

from fakes import FakeClient


def transcript(p):
    return " ".join(m["content"] for s in p.sessions for m in s.messages).lower()


def test_dataset_is_deterministic_and_answers_are_in_the_conversations():
    a, b = make_persona(3), make_persona(3)
    assert [s.messages for s in a.sessions] == [s.messages for s in b.sessions]
    for p in dataset(6):
        text = transcript(p)
        for q in p.questions:
            for group in q.expected:
                assert any(alias in text for alias in group), (p.name, q.id)
        assert "sister" not in text and " car " not in text  # abstention questions have no answer
        assert p.facts["friend_pet"].lower() in text  # the distractor is there


def test_filler_sessions_add_history_without_new_facts():
    p = make_persona(1, filler_sessions=10)
    assert len(p.sessions) == 18
    assert transcript(p).count(p.facts["pet"].lower()) == transcript(make_persona(1)).count(p.facts["pet"].lower())


def test_scoring():
    q = {x.id.split(":")[1]: x for x in make_persona(0).questions}
    assert is_correct(q["countries"], True, "Peru, Iceland and Kenya.")
    assert not is_correct(q["countries"], True, "Peru and Kenya")
    assert is_correct(q["race_date"], True, "Saturday, August 29th")
    assert is_correct(q["sister"], False, "I don't know")
    assert not is_correct(q["sister"], True, "Maya")
    assert not is_correct(q["pet_name"], False, "Luna?")
    assert normalize("Head-Nurse!") == "head-nurse"


def test_chunk_rag_finds_the_distractor_too():
    p = make_persona(0)
    ctx = chunk_context(p, "What's my pet called?")
    assert p.facts["pet"] in ctx or p.facts["friend_pet"] in ctx


def test_runner_end_to_end_with_fake_model():
    def script(kw):
        if kw["system"] == EXTRACT_SYSTEM:
            return {"memories": []}
        return {"known": False, "answer": "I don't know", "memory_ids": []}

    r = Runner("claude-haiku-5-5", client=FakeClient(script), verbose=False)
    results = r.run(dataset(1), ["no_memory", "memory"])
    assert len(results) == 24
    assert sum(x.correct for x in results) == 4  # only the 2 abstention questions, per method
    table = summarize(results, ["no_memory", "memory"], "claude-haiku-5-5", r.ingest_usage)
    assert "| no_memory | **17%** |" in table and "Memory building: 8 calls" in table
