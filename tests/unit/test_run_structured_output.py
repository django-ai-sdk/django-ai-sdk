from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from django_ai_sdk.adapters.base import Run
from django_ai_sdk.common import ChatMessage


class Extraction(BaseModel):
    name: str
    count: int


class Listing(BaseModel):
    """Has a list, so a truncated reply can still validate — just shorter."""

    title: str
    items: list[str]


class FakeGenerator:
    """Returns a fixed reply, standing in for a Haystack chat generator."""

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls = 0

    def run(self, messages, generation_kwargs=None):
        self.calls += 1
        reply = SimpleNamespace(text=self.text, meta={"finish_reason": self.finish_reason})
        return {"replies": [reply]}


def _messages():
    return [ChatMessage(role="user", content="extract this")]


class TestRunStructuredOutput:
    async def test_parses_well_formed_json(self):
        generator = FakeGenerator('{"name": "widget", "count": 3}')
        run = Run(generator)

        result = await run.run(_messages(), response_format=Extraction)

        assert result == Extraction(name="widget", count=3)
        assert generator.calls == 1

    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "widget", "count": 3}}',  # stray trailing brace
            '```json\n{"name": "widget", "count": 3}\n```',  # fenced
            'Here you go: {"name": "widget", "count": 3}',  # prose preamble
            "{'name': 'widget', 'count': 3}",  # single quotes
        ],
    )
    async def test_repairs_output_a_strict_parse_rejects(self, text):
        """Models reached through an OpenAI-compatible endpoint do not reliably
        honour json_schema, so the strict parse alone loses usable answers."""
        run = Run(FakeGenerator(text))

        result = await run.run(_messages(), response_format=Extraction)

        assert result == Extraction(name="widget", count=3)

    async def test_still_raises_when_there_is_nothing_to_repair(self):
        """Repair is a second chance, not a guarantee — a reply with none of the
        expected fields must not be silently turned into a default model."""
        run = Run(FakeGenerator("the model refused to answer"))

        with pytest.raises(ValidationError):
            await run.run(_messages(), response_format=Extraction)

    async def test_returns_plain_text_without_a_response_format(self):
        run = Run(FakeGenerator("just prose"))

        assert await run.run(_messages()) == "just prose"


class TestRunTruncatedOutput:
    """Truncation must not be repaired.

    Cut a list mid-way and the repair closes it, so the result validates while
    silently missing whatever came after the cut. For an extraction persisted to
    a knowledge base, quietly indexing half a document is worse than failing:
    a failure can be retried, a plausible-looking partial cannot even be noticed.
    """

    # Ends inside the list, on a clean element boundary — what a max_tokens
    # cutoff produces.
    TRUNCATED = '{"title": "Report", "items": ["alpha", "beta"'

    async def test_truncated_output_raises_instead_of_repairing(self):
        run = Run(FakeGenerator(self.TRUNCATED, finish_reason="length"))

        with pytest.raises(ValidationError):
            await run.run(_messages(), response_format=Listing)

    async def test_the_same_text_is_repaired_when_not_truncated(self):
        """The pair that matters: identical text, and only finish_reason differs.

        Proves the gate is doing the work — without it this text repairs into a
        clean object, which is exactly the silent loss being prevented above.
        """
        run = Run(FakeGenerator(self.TRUNCATED, finish_reason="stop"))

        result = await run.run(_messages(), response_format=Listing)

        assert result == Listing(title="Report", items=["alpha", "beta"])

    async def test_repairs_when_the_provider_reports_no_finish_reason(self):
        """Not every provider populates meta; absence must not block the repair."""

        class NoMetaGenerator:
            def run(self, messages, generation_kwargs=None):
                return {"replies": [SimpleNamespace(text=self.text, meta=None)]}

        generator = NoMetaGenerator()
        generator.text = '{"title": "Report", "items": ["alpha"]}}'

        result = await Run(generator).run(_messages(), response_format=Listing)

        assert result == Listing(title="Report", items=["alpha"])
