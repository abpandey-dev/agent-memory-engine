"""Thin wrapper for structured-output calls to Claude, with token and cost accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallbacks ("default" mode) are only available on these models.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
# USD per million tokens (input, output).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def cost(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        p_in, p_out = PRICES[model]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.usage = Usage()

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict:
        extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.model in FALLBACK_MODELS else {}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **extra,
        )
        self.usage.add(Usage(1, response.usage.input_tokens, response.usage.output_tokens))
        if response.stop_reason == "refusal":
            raise LLMError("Model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("Response hit max_tokens before finishing")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Could not parse JSON output: {e}") from e
