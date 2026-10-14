import asyncio
import json

from mcp import Client

from agent_memory import mcp_server
from agent_memory.engine import MemoryEngine
from agent_memory.llm import LLM

from fakes import FakeClient, mem


def test_mcp_tools_remember_recall_forget():
    client = FakeClient(lambda kw: {"memories": [mem("The user's cat is called Mochi.", ["cat", "pet", "kitten"],
                                                     [("Mochi", "pet")])]})
    mcp_server.set_engine(MemoryEngine(llm=LLM("claude-haiku-5-5", client=client)))

    async def run():
        async with Client(mcp_server.server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"remember", "recall", "list_memories", "list_entities", "forget"}
            added = await c.call_tool("remember", {"text": "Our cat is Mochi", "user_id": "u"})
            recalled = await c.call_tool("recall", {"query": "what is my pet's name", "user_id": "u"})
            ents = await c.call_tool("list_entities", {"user_id": "u"})
            gone = await c.call_tool("forget", {"memory_id": "m-0001"})
            left = await c.call_tool("list_memories", {"user_id": "u"})
            return added, recalled, ents, gone, left

    added, recalled, ents, gone, left = asyncio.run(run())
    assert added.structured_content["added"] == ["m-0001"]
    assert "Mochi" in json.dumps(recalled.structured_content)
    assert "Mochi" in json.dumps(ents.structured_content)
    assert gone.structured_content["deleted"] is True
    assert "Mochi" not in json.dumps(left.structured_content)
