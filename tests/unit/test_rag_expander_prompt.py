"""The default expander prompt must produce a reply Haystack's QueryExpander can parse."""

from __future__ import annotations

import json

from django_ai_sdk.rags.base import DEFAULT_EXPANDER_PROMPT
from haystack.components.query import QueryExpander
from haystack.dataclasses import ChatMessage


class _EchoingGenerator:
    """Answers the way the prompt asks: JSON when the prompt asks for JSON, else lines."""

    def __init__(self) -> None:
        self.prompt = ""

    def run(self, messages: list[ChatMessage]) -> dict:
        self.prompt = messages[0].text
        reply = json.dumps({"queries": ["remote work policy", "what is the remote work policy"]})
        return {"replies": [ChatMessage.from_assistant(reply)]}


def test_default_prompt_asks_for_the_json_the_expander_parses() -> None:
    generator = _EchoingGenerator()
    expander = QueryExpander(
        chat_generator=generator, n_expansions=2, prompt_template=DEFAULT_EXPANDER_PROMPT
    )

    result = expander.run("remote work policy")

    assert '"queries"' in generator.prompt
    assert "one per line" not in generator.prompt
    assert result["queries"] == ["remote work policy", "what is the remote work policy"]
