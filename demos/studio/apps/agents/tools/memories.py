from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.artifacts import FileArtifact, ToolArtifact
from django.db.models import Q
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, EntryDocument
from django_ai_sdk.memories.tools import ASK_IMAGE_ARTIFACT, ASK_IMAGE_TOOL, get_thread_file
from haystack.tools import Tool

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from django_ai_sdk.common import ChatMessage

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


def list_memory_files(thread_id: str, keywords: list[str] | None = None) -> list[dict] | dict:
    """All files, or with ``keywords`` those whose name, summary, keywords or facts
    contain every term (partial, case-insensitive), with their summary and keywords."""
    entries = Entry.objects.for_rag(thread_id)
    terms = [t.strip() for t in keywords or [] if t and t.strip()]
    if not terms:
        return [info for _, info in get_files(entries, thread_id)]
    for term in terms:
        # keywords and facts are JSON lists, matched as their text
        entries = entries.filter(
            Q(name__icontains=term)
            | Q(data__summary__icontains=term)
            | Q(data__keywords__icontains=term)
            | Q(data__facts__icontains=term)
        )
    files = []
    for entry, info in get_files(entries, thread_id):
        extraction = entry.extraction
        files.append(
            info
            | {
                "summary": extraction.summary if extraction else "",
                "keywords": extraction.keywords if extraction else [],
            }
        )
    if not files:
        return {
            "error": f"No file matches all of {terms}. Retry with fewer or shorter keywords, "
            "or search the knowledge bases with the search tools instead."
        }
    return {
        "files": files,
        # Read when picking the next step: a list alone gives nothing to cite
        "next": "Search the knowledge bases with these files' keywords for the "
        "details, and answer from both: this list for completeness, the passages "
        "to cite.",
    }


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
            "error": f"No file named '{filename}' in this thread. If this is a topic rather "
            "than a file name, search the knowledge bases with the search tools instead; "
            "otherwise tell the user the file is not available, without describing its "
            "contents."
        }
    return files


def get_memory_files(thread_id: str = "", **kwargs: object) -> Tool:
    return Tool(
        name="get_files",
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
            "For 'all / which / list files about X' questions, pass short keyword "
            "stems in `keywords` (e.g. ['lesson', '2026']) instead of searching: "
            "it returns every file whose name, summary, keywords or facts contain "
            "all of them, with its summary and keywords. Then search inside the "
            "matched files for details. "
            "If the user asked about a file's content (what's in it, summarise "
            "it, etc.), listing alone does not answer that: call "
            "get_file with its name, then search that file's memory "
            "using what get_file found (keywords, facts, entities) as "
            "the query. Only stop at the listing "
            "itself when the user asked to enumerate files, not about their "
            "content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Short stems a file must all contain "
                    "(partial, case-insensitive). Omit to list every file.",
                }
            },
            "required": [],
        },
        function=lambda keywords=None, **_: list_memory_files(thread_id, keywords),
    )


def get_memory_file(thread_id: str = "", **kwargs: object) -> Tool:
    return Tool(
        name="get_file",
        description=(
            "Look up a file in the current thread by name and return what it is "
            "about: summary, keywords, facts, entities and a content preview, "
            "plus the same fields as get_files. Use this when the user "
            "names a file ('what is in xxx.pdf?'). Not for topics or keywords: "
            "to find what documents say, use the search "
            "tools. After the lookup, search the file's memory "
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


async def get_looked_up_files(
    arguments: dict[str, Any],
    result: list[dict[str, Any]] | dict[str, Any],
    thread_id: str,
) -> dict[str, list[dict[str, str]]] | None:
    """The thread uploads get_memory_file found"""
    if not isinstance(result, list):
        return None
    entry_ids = [f["entry_id"] for f in result if f["source"] == "attachment"]
    docs = EntryDocument.objects.filter(entry_id__in=entry_ids, memory__thread_files__id=thread_id)
    files = [get_thread_file(doc, thread_id) async for doc in docs]
    return {"files": files} if files else None


class FileLookupMixin:
    """A turn with an attached file looks it up with get_memory_file first"""

    tool_artifacts = {
        ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT,
        # here is different syntax: for demo
        "get_memory_file": ToolArtifact(FileArtifact, get_looked_up_files),
    }

    def get_run_required_tools(self, messages: list[ChatMessage]) -> list[str]:
        """Names of tools the model must have called before it answers this turn."""
        last = next((m for m in reversed(messages) if m.role == "user"), None)
        if last and any(not a.media_type.startswith("image/") for a in last.attachments):
            return ["get_memory_file"]
        return []
