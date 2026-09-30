from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry
from haystack.tools import Tool

if TYPE_CHECKING:
    from django.db.models import QuerySet

# Enough to show what a file is about; search the memory for the rest.
PREVIEW_CHARS = 2000


def get_files(entries: QuerySet[Entry], thread_id: str) -> list[tuple[Entry, dict]]:
    """Each entry with its listing fields, newest first."""
    file_memory_id = (
        Thread.objects.filter(id=thread_id).values_list("file_memory_id", flat=True).first()
    )
    return [
        (
            entry,
            {
                "entry_id": str(entry.id),
                "filename": entry.name,
                "memory_name": entry.memory.name,
                "memory_slug": entry.memory.slug,
                "file_size": entry.file_size,
                # Thread uploads vs linked knowledge bases
                "source": "attachment" if entry.memory_id == file_memory_id else "knowledge_base",
                "created_at": entry.created_at.isoformat(),
            },
        )
        for entry in entries.order_by("-created_at")
    ]


def list_memory_files(thread_id: str) -> list[dict]:
    return [info for _, info in get_files(Entry.objects.for_rag(thread_id), thread_id)]


def list_memory_file(thread_id: str, filename: str) -> list[dict] | dict:
    """Files matching ``filename`` (partial, case-insensitive), with their details."""
    entries = Entry.objects.for_rag(thread_id).filter(name__icontains=filename)
    files = []
    for entry, info in get_files(entries, thread_id):
        extraction = entry.extraction
        files.append(
            info
            | {
                "summary": extraction.summary if extraction else "",
                "keywords": extraction.keywords if extraction else [],
                "facts": [fact.text for fact in extraction.facts] if extraction else [],
                "entities": [f"{e.text} ({e.type})" for e in extraction.entities]
                if extraction
                else [],
                "preview": entry.content[:PREVIEW_CHARS],
            }
        )
    if not files:
        # An empty list is easy to gloss over; say plainly the file isn't there.
        return {
            "error": f"No file matching '{filename}' in this thread. Tell the user it is "
            "not available; do not describe its contents."
        }
    return files


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
            "handle a request. "
            "If the user asked about a file's content (what's in it, summarise "
            "it, etc.), listing alone does not answer that: call "
            "get_memory_file with its name, then search that file's memory "
            "using what get_memory_file found (keywords, facts, entities) as "
            "the query. Only stop at the listing "
            "itself when the user asked to enumerate files, not about their "
            "content."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda: list_memory_files(thread_id),
    )


def get_memory_file(thread_id: str = "", **kwargs: object) -> Tool:
    return Tool(
        name="get_memory_file",
        description=(
            "Look up a file in the current thread by name and return what it is "
            "about: summary, keywords, facts, entities and a content preview, "
            "plus the same fields as get_memory_files. Use this when the user "
            "names a file ('what is in cv.pdf?'). Then search the file's memory "
            "(source 'attachment': the uploaded-documents search tool) using its "
            "keywords, facts or entities as the query, and answer from the "
            "search results so the answer can cite them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Full or partial filename, case-insensitive.",
                }
            },
            "required": ["filename"],
        },
        function=lambda filename: list_memory_file(thread_id, filename),
    )
