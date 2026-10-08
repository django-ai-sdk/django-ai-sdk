"""get_files narrows the file list by keywords over name, summary, keywords and facts."""

from __future__ import annotations

import pytest
from asgiref.sync import async_to_sync
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, Memory, ThreadMemory
from django_ai_sdk.memories.schemas import Predicate

from apps.agents.tools import memories

list_memory_files = async_to_sync(memories.list_memory_files)


def _extraction(summary: str, keywords: list[str], facts: tuple[str, ...] = ()) -> dict:
    return {
        "summary": summary,
        "keywords": list(keywords),
        "entities": [],
        "facts": [
            {
                "subject": "",
                "predicate": list(Predicate)[0],
                "object": "",
                "text": f,
                "evidence": "",
            }
            for f in facts
        ],
        "sections": [],
        "events": [],
    }


@pytest.mark.django_db
def test_keywords_filter_files() -> None:
    thread = Thread.objects.create()
    memory = Memory.objects.create(name="Lessons")
    ThreadMemory.objects.create(thread=thread, memory=memory)
    Entry.objects.create(
        memory=memory, name="ll-01.pdf", data=_extraction("Lessons learned 2026", ["safety"])
    )
    Entry.objects.create(
        memory=memory,
        name="ll-02.pdf",
        data=_extraction("Incident review", ["lesson"], ["Happened in 2026"]),
    )
    Entry.objects.create(
        memory=memory, name="ll-03.pdf", data=_extraction("Lessons learned 2025", ["safety"])
    )
    tid = str(thread.id)

    def names(result: list[dict] | dict) -> list[str]:
        files = result["files"] if isinstance(result, dict) else result
        return sorted(f["filename"] for f in files)

    assert names(list_memory_files(tid)) == ["ll-01.pdf", "ll-02.pdf", "ll-03.pdf"]
    # Every term must match, each in any field; "lesson" stems "Lessons"
    assert names(list_memory_files(tid, ["lesson", "2026"])) == ["ll-01.pdf", "ll-02.pdf"]
    assert names(list_memory_files(tid, ["LL-03"])) == ["ll-03.pdf"]
    assert list_memory_files(tid, ["2026"])["files"][0]["summary"]
    assert "next" in list_memory_files(tid, ["2026"])
    assert "error" in list_memory_files(tid, ["lesson", "2024"])
