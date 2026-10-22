"""Regex per line over memory text, decided in the database (SQLite here)."""

import pytest
from django_ai_sdk.memories.expressions import LineRegex, agrep, get_pattern
from django_ai_sdk.memories.models import Entry, Memory

REPORT = "# Incident INC-2024-05-001\nSpill at the loading dock.\nline 3"


@pytest.mark.django_db
@pytest.mark.asyncio
class TestLineRegex:
    async def _entries(self):
        memory = await Memory.objects.acreate(name="KB")
        await Entry.objects.acreate(memory=memory, name="incident.md", content=REPORT)
        await Entry.objects.acreate(memory=memory, name="other.md", content="nothing here")
        return Entry.objects.filter(memory=memory)

    async def test_only_rows_with_a_matching_line(self):
        entries = await self._entries()

        names = [e.name async for e in entries.filter(LineRegex("content", "inc-2024"))]
        case_sensitive = entries.filter(LineRegex("content", "inc-2024", ignore_case=False))

        assert names == ["incident.md"]
        assert not await case_sensitive.aexists()

    async def test_anchors_match_at_line_ends(self):
        entries = await self._entries()

        found = entries.filter(LineRegex("content", r"^Spill.*dock\.$"))

        assert [e.name async for e in found] == ["incident.md"]

    async def test_agrep_yields_each_matching_line_with_its_number(self):
        entries = await self._entries()

        hits = [(e.name, n, line) async for e, n, line in agrep(entries, get_pattern("dock|line"))]

        assert hits == [
            ("incident.md", 2, "Spill at the loading dock."),
            ("incident.md", 3, "line 3"),
        ]


def test_an_invalid_regex_is_literal_text():
    assert get_pattern("INC-2024(").search("see INC-2024( here")
    assert get_pattern("a.c").search("abc")  # a valid one stays a regex
