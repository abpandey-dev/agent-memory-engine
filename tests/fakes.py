"""A scripted stand-in for the Anthropic client: no network, no cost."""

import json
from types import SimpleNamespace as NS


class FakeClient:
    """`handler(kwargs) -> dict` decides the JSON each call returns."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.handler(kwargs)
        if out == "refusal":
            return NS(content=[], stop_reason="refusal", usage=NS(input_tokens=10, output_tokens=0))
        return NS(content=[NS(type="text", text=json.dumps(out))], stop_reason="end_turn",
                  usage=NS(input_tokens=100, output_tokens=20))


def mem(text, keywords=(), entities=(), kind="fact", importance=3):
    return {"text": text, "kind": kind, "keywords": list(keywords),
            "entities": [{"name": n, "type": t} for n, t in entities], "importance": importance}
