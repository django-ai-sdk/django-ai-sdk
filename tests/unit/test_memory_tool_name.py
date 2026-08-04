from __future__ import annotations

import re
import uuid

import pytest


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryToolName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Legal Documents", "search_legal_documents"),
            ("Hello?", "search_hello"),
            ("Q&A", "search_q_a"),
            ("Sales (EU)", "search_sales_eu"),
            ("HR / Legal", "search_hr_legal"),
            ("  spaced  out  ", "search_spaced_out"),
            ("--dashes--", "search_dashes"),
            # Truncation must not leave a trailing separator behind.
            ("abcdefghijklmnopqrs tuv", "search_abcdefghijklmnopqrs"),
        ],
    )
    async def test_sanitises_name(self, name, expected):
        from django_ai_sdk.memories.models import Memory

        memory = await Memory.objects.acreate(name=name)
        assert memory.tool_name == expected

    @pytest.mark.parametrize("name", ["???", "日本語", "—"])
    async def test_falls_back_to_id_when_nothing_survives(self, name):
        from django_ai_sdk.memories.models import Memory

        memory = await Memory.objects.acreate(name=name)
        assert memory.tool_name == f"search_memory_{str(memory.id).replace('-', '')[:8]}"

    async def test_name_is_always_valid(self):
        from django_ai_sdk.memories.models import Memory

        for name in ["Hello?", "Q&A", "???", "日本語", "a" * 300, "!!! ### ???"]:
            tool_name = Memory(id=uuid.uuid4(), name=name).tool_name
            assert re.fullmatch(r"[a-zA-Z0-9_-]+", tool_name), tool_name
