from __future__ import annotations

from django.db.models import Case, CharField, Subquery, Value, When
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry
from haystack.tools import Tool


def list_memory_files(thread_id: str) -> list[dict]:
    entries = (
        Entry.objects.for_rag(thread_id)
        .annotate(
            # annotate source from query
            source=Case(
                When(
                    memory_id=Subquery(
                        Thread.objects.filter(id=thread_id).values("file_memory_id")[:1]
                    ),
                    then=Value("attachment"),
                ),
                default=Value("knowledge_base"),
                output_field=CharField(),
            )
        )
        .order_by("-created_at")
    )

    return [
        {
            "entry_id": str(entry.id),
            "filename": entry.name,
            "memory_name": entry.memory.name,
            "memory_slug": entry.memory.slug,
            "file_size": entry.file_size,
            "source": entry.source,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in entries
    ]


def get_memory_files(thread_id: str = "", **kwargs: object) -> Tool:
    return Tool(
        name="get_memory_files",
        description=(
            "List files available in the current thread (knowledge base + "
            "attachments), newest first. Returns entry_id, filename, "
            "memory_name, memory_slug, file_size (bytes for uploads, "
            "characters for text-only entries), source ('attachment' for "
            "uploads vs 'knowledge_base'), and created_at for each. Use this "
            "to enumerate available documents — "
            "for example to answer 'what files are available?', to discover "
            "an entry_id needed by another tool (e.g. search_memory), or to "
            "check whether a specific file is present before deciding how to "
            "handle a request. If the user asked about a file's content (what's "
            "in it, summarise it, etc.), listing alone does not answer that — "
            "follow up with the uploaded-documents search tool for that file's "
            "memory. Only stop at the listing itself when the user asked to "
            "enumerate files, not about their content."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda: list_memory_files(thread_id),
    )
