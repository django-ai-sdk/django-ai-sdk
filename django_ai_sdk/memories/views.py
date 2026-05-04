"""
TODO: we should use this as a reference only, for now there is no authentication.
"""

import os

from django.db.models import Count
from django.http import HttpRequest
from ninja import File, Router
from ninja.files import UploadedFile

from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, EntryDocument, Memory, ThreadMemory
from django_ai_sdk.memories.schemas import (
    BulkConnectMemoriesIn,
    DocumentOut,
    MemoryIn,
    MemoryOut,
    ThreadMemoryOut,
    ToggleMemoryActiveIn,
)
from django_ai_sdk.memories.utils import extract_document

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}

router = Router()


def _read_text_content(file: UploadedFile) -> tuple[str, str] | None:
    """Read text content from a file. Returns (content, ext) or None if unsupported/empty."""
    file_name = file.name or ""
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    if ext not in SUPPORTED_TEXT_EXTENSIONS:
        return None

    content = file.read().decode("utf-8", errors="replace")
    if not content.strip():
        return None

    return content, ext


@router.post("", response=MemoryOut)
async def create_memory(request: HttpRequest, payload: MemoryIn) -> MemoryOut:
    """Create a new memory."""
    memory = await Memory.objects.acreate(
        name=payload.name,
        description=payload.description or "",
    )
    return MemoryOut(
        id=str(memory.id),
        name=memory.name,
        description=memory.description,
        document_count=0,
        created_at=memory.created_at.isoformat(),
        updated_at=memory.updated_at.isoformat(),
    )


@router.get("", response=list[MemoryOut])
def list_memories(request: HttpRequest) -> list[MemoryOut]:
    """List all visible memories."""
    memories = (
        Memory.objects.filter(is_hidden=False)
        .annotate(document_count=Count("entries"))
        .order_by("-created_at")
    )
    return [
        MemoryOut(
            id=str(memory.id),
            name=memory.name,
            description=memory.description,
            document_count=memory.document_count,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )
        for memory in memories
    ]


@router.get("/{memory_id}", response=MemoryOut)
async def get_memory(request: HttpRequest, memory_id: str) -> MemoryOut:
    """Get a single memory by ID."""
    memory = await Memory.objects.annotate(document_count=Count("entries")).aget(id=memory_id)
    return MemoryOut(
        id=str(memory.id),
        name=memory.name,
        description=memory.description,
        document_count=memory.document_count,
        created_at=memory.created_at.isoformat(),
        updated_at=memory.updated_at.isoformat(),
    )


@router.put("/{memory_id}", response=MemoryOut)
def update_memory(request: HttpRequest, memory_id: str, payload: MemoryIn) -> MemoryOut:
    """Update a memory."""
    memory = Memory.objects.get(id=memory_id)
    memory.name = payload.name
    memory.description = payload.description or ""
    memory.save()
    doc_count = memory.entries.count()
    return MemoryOut(
        id=str(memory.id),
        name=memory.name,
        description=memory.description,
        document_count=doc_count,
        created_at=memory.created_at.isoformat(),
        updated_at=memory.updated_at.isoformat(),
    )


@router.delete("/{memory_id}", response={204: None})
async def delete_memory(request: HttpRequest, memory_id: str) -> tuple[int, None]:
    """Delete a memory and all its entries."""
    memory = await Memory.objects.aget(id=memory_id)
    await memory.adelete()
    return 204, None


@router.post("/{memory_id}/documents", response={200: DocumentOut, 400: dict})
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut | tuple[int, dict]:
    """Upload a file to a memory."""

    memory = await Memory.objects.aget(id=memory_id)
    file_name = file.name or ""

    result = _read_text_content(file)
    if result is None:
        return 400, {"detail": f"Unsupported or empty file: {file_name}"}
    content, ext = result

    extraction = await extract_document(content)

    entry = await Entry.objects.acreate(
        memory=memory,
        name=file_name,
        content=content,
        data=extraction.model_dump() if extraction else {},
    )

    entry_doc = await EntryDocument.objects.acreate(
        entry=entry,
        file=file,
        file_name=file_name,
        file_size=file.size or 0,
        content_type=file.content_type or "",
        file_extension=ext.lstrip("."),
        extracted=bool(extraction),
    )

    return DocumentOut(
        id=str(entry.id),
        file=entry_doc.file.url if entry_doc.file else "",
        content=entry.content,
        extraction=entry.extraction,
        file_name=entry_doc.file_name,
        file_size=entry_doc.file_size,
        content_type=entry_doc.content_type,
        file_extension=entry_doc.file_extension,
        created_at=entry_doc.created_at.isoformat(),
        updated_at=entry_doc.updated_at.isoformat(),
    )


def _entry_doc_to_out(entry_doc: EntryDocument) -> DocumentOut:
    entry = entry_doc.entry
    return DocumentOut(
        id=str(entry.id),
        file=entry_doc.file.url if entry_doc.file else "",
        content=entry.content,
        extraction=entry.extraction,
        file_name=entry_doc.file_name,
        file_size=entry_doc.file_size,
        content_type=entry_doc.content_type,
        file_extension=entry_doc.file_extension,
        created_at=entry_doc.created_at.isoformat(),
        updated_at=entry_doc.updated_at.isoformat(),
    )


@router.get("/{memory_id}/documents", response=list[DocumentOut])
def list_documents(request: HttpRequest, memory_id: str) -> list[DocumentOut]:
    """List all file-backed documents in a memory."""
    entry_docs = (
        EntryDocument.objects.filter(entry__memory_id=memory_id)
        .select_related("entry")
        .order_by("-created_at")
    )
    return [_entry_doc_to_out(ed) for ed in entry_docs]


@router.get("/{memory_id}/documents/{doc_id}", response=DocumentOut)
def get_document(request: HttpRequest, memory_id: str, doc_id: str) -> DocumentOut:
    """Get a single document from a memory."""
    entry_doc = EntryDocument.objects.select_related("entry").get(
        entry_id=doc_id, entry__memory_id=memory_id
    )
    return _entry_doc_to_out(entry_doc)


@router.delete("/{memory_id}/documents/{doc_id}", response={204: None})
async def delete_document(request: HttpRequest, memory_id: str, doc_id: str) -> tuple[int, None]:
    """Delete a document (and its entry) from a memory."""
    entry = await Entry.objects.aget(id=doc_id, memory_id=memory_id)
    await entry.adelete()
    return 204, None


@router.post("/{memory_id}/link/{thread_id}", response={204: None})
async def link_thread(request: HttpRequest, memory_id: str, thread_id: str) -> tuple[int, None]:
    """Link a memory to a thread."""
    memory = await Memory.objects.aget(id=memory_id)

    thread = await Thread.objects.aget(id=thread_id)
    await ThreadMemory.objects.aget_or_create(
        thread=thread,
        memory=memory,
    )
    return 204, None


@router.delete("/{memory_id}/link/{thread_id}", response={204: None})
async def unlink_thread(request: HttpRequest, memory_id: str, thread_id: str) -> tuple[int, None]:
    """Unlink a memory from a thread."""
    link = await ThreadMemory.objects.aget(memory_id=memory_id, thread_id=thread_id)
    await link.adelete()
    return 204, None


# ============================================================================
# Thread-Memory Management Endpoints
# ============================================================================


@router.get("/thread/{thread_id}", response=list[ThreadMemoryOut])
async def list_thread_memories(request: HttpRequest, thread_id: str) -> list[ThreadMemoryOut]:
    """List all memories connected to a thread with their active status."""
    thread_memories_query = (
        ThreadMemory.objects.filter(thread_id=thread_id, memory__is_hidden=False)
        .select_related("memory")
        .annotate(document_count=Count("memory__entries"))
    )

    memories = []
    async for tm in thread_memories_query:
        memories.append(
            ThreadMemoryOut(
                id=str(tm.memory.id),
                name=tm.memory.name,
                description=tm.memory.description,
                document_count=tm.document_count,
                active=tm.active,
                created_at=tm.created_at.isoformat(),
            )
        )

    return memories


@router.post("/thread/{thread_id}/bulk", response=list[ThreadMemoryOut])
async def bulk_connect_memories(
    request: HttpRequest, thread_id: str, payload: BulkConnectMemoriesIn
) -> list[ThreadMemoryOut]:
    """Connect multiple memories to a thread at once."""

    thread = await Thread.objects.aget(id=thread_id)

    for memory_id in payload.memory_ids:
        memory = await Memory.objects.aget(id=memory_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
            defaults={"active": True},
        )

    return await list_thread_memories(request, thread_id)


# ============================================================================
# Thread File Upload Endpoints
# ============================================================================


async def _get_or_create_thread_file_memory(thread_id: str) -> Memory:
    """Get or auto-create the hidden file-upload Memory for a thread."""

    thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
    if not thread.file_memory:
        memory = await Memory.objects.acreate(
            name=f"thread_files_{thread_id}",
            description="Thread file uploads",
            is_hidden=True,
        )
        thread.file_memory = memory
        await thread.asave(update_fields=["file_memory", "updated_at"])
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)
    return thread.file_memory


@router.post("/thread/{thread_id}/files", response={200: DocumentOut, 400: dict})
async def upload_thread_file(
    request: HttpRequest,
    thread_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut | tuple[int, dict]:
    """Upload a file to a thread. Auto-creates a hidden memory on first upload."""

    memory = await _get_or_create_thread_file_memory(thread_id)
    file_name = file.name or ""

    result = _read_text_content(file)
    if result is None:
        return 400, {"detail": f"Unsupported or empty file: {file_name}"}
    content, ext = result

    extraction = await extract_document(content)

    entry = await Entry.objects.acreate(
        memory=memory,
        name=file_name,
        content=content,
        data=extraction.model_dump() if extraction else {},
    )

    entry_doc = await EntryDocument.objects.acreate(
        entry=entry,
        file=file,
        file_name=file_name,
        file_size=file.size or 0,
        content_type=file.content_type or "",
        file_extension=ext.lstrip("."),
        extracted=bool(extraction),
    )

    return DocumentOut(
        id=str(entry.id),
        file=entry_doc.file.url if entry_doc.file else "",
        content=entry.content,
        extraction=entry.extraction,
        file_name=entry_doc.file_name,
        file_size=entry_doc.file_size,
        content_type=entry_doc.content_type,
        file_extension=entry_doc.file_extension,
        created_at=entry_doc.created_at.isoformat(),
        updated_at=entry_doc.updated_at.isoformat(),
    )


@router.get("/thread/{thread_id}/files", response=list[DocumentOut])
def list_thread_files(request: HttpRequest, thread_id: str) -> list[DocumentOut]:
    """List all files uploaded to a thread."""
    try:
        thread = Thread.objects.select_related("file_memory").get(id=thread_id)
    except Thread.DoesNotExist:
        return []

    if not thread.file_memory_id:
        return []

    entry_docs = (
        EntryDocument.objects.filter(entry__memory_id=thread.file_memory_id)
        .select_related("entry")
        .order_by("-created_at")
    )
    return [_entry_doc_to_out(ed) for ed in entry_docs]


@router.delete("/thread/{thread_id}/files/{doc_id}", response={204: None})
async def delete_thread_file(request: HttpRequest, thread_id: str, doc_id: str) -> tuple[int, None]:
    """Delete a file from a thread."""

    thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
    if not thread.file_memory_id:
        return 204, None
    entry = await Entry.objects.aget(id=doc_id, memory_id=thread.file_memory_id)
    await entry.adelete()
    return 204, None


# ============================================================================
# Thread-Memory Management Endpoints
# ============================================================================


@router.patch("/thread/{thread_id}/{memory_id}", response=ThreadMemoryOut)
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> ThreadMemoryOut:
    """Toggle the active status of a memory for a thread."""
    thread_memory = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
    thread_memory.active = payload.active
    await thread_memory.asave()

    memory = await Memory.objects.aget(id=memory_id)
    doc_count = await Entry.objects.filter(memory_id=memory_id).acount()

    return ThreadMemoryOut(
        id=str(memory.id),
        name=memory.name,
        description=memory.description,
        document_count=doc_count,
        active=thread_memory.active,
        created_at=thread_memory.created_at.isoformat(),
    )


@router.delete("/thread/{thread_id}/{memory_id}", response={204: None})
async def disconnect_memory_from_thread(
    request: HttpRequest, thread_id: str, memory_id: str
) -> tuple[int, None]:
    """Disconnect (delete) a memory from a thread."""
    link = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
    await link.adelete()
    return 204, None
