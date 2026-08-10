from __future__ import annotations

from django_ai_sdk.memories.models import Entry, ThreadMemory
from haystack.tools import Tool


def list_memory_files(thread_id: str) -> list[dict]:
    files = []
    for link in ThreadMemory.objects.filter(thread_id=thread_id, active=True).select_related(
        "memory"
    ):
        files.extend(
            [
                {
                    "entry_id": str(entry.id),
                    "filename": entry.name,
                    "memory_name": link.memory.name,
                    "file_size_chars": len(entry.content or ""),
                }
                for entry in Entry.objects.filter(memory=link.memory).order_by("created_at")
            ]
        )
    return files


def get_memory_files(thread_id: str = "", **kwargs: object) -> Tool:
    return Tool(
        name="get_memory_files",
        description=(
            "List files available in the current thread (knowledge base + "
            "attachments). Returns entry_id, filename, memory and size for each. "
            "Use this to enumerate available documents — for example to answer "
            "'what files are available?', to discover an entry_id needed by "
            "another tool (e.g. search_memory), or to check whether a "
            "specific file is present before deciding how to handle a request. "
            "Listing is a standalone action: report the result and stop. Do not "
            "chain into retrieval, summarisation, or translation as a "
            "follow-up step on your own initiative."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda: list_memory_files(thread_id),
    )
