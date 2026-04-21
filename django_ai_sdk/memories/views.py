"""
TODO: we should use this as a reference only, for now there is no authentication.
"""

from django.db.models import Count
from django.http import HttpRequest
from ninja import File, Router
from ninja.files import UploadedFile

from django_ai_sdk.memories.models import Document, Memory, ThreadMemory
from django_ai_sdk.memories.schemas import (
    BulkConnectMemoriesIn,
    DocumentOut,
    MemoryIn,
    MemoryOut,
    ThreadMemoryOut,
    ToggleMemoryActiveIn,
)

router = Router()


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
    """List all memories."""
    memories = Memory.objects.annotate(document_count=Count("documents")).order_by("-created_at")
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
    memory = await Memory.objects.annotate(document_count=Count("documents")).aget(id=memory_id)
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
    doc_count = memory.documents.count()
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
    """Delete a memory and all its documents."""
    memory = await Memory.objects.aget(id=memory_id)
    await memory.adelete()
    return 204, None


@router.post("/{memory_id}/documents", response=DocumentOut)
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut:
    """Upload a file to a memory."""
    import os

    from django_ai_sdk.memories.utils import extract_document

    memory = await Memory.objects.aget(id=memory_id)
    file_name = file.name or ""
    _, ext = os.path.splitext(file_name)

    content = ""
    extraction = None
    if ext.lower() in (".txt", ".md"):
        content = file.read().decode("utf-8", errors="replace")
        extraction = await extract_document(content)

    doc = await Document.objects.acreate(
        memory=memory,
        file=file,
        content=content,
        file_name=file.name,
        file_size=file.size or 0,
        content_type=file.content_type or "",
        file_extension=ext.lower().lstrip("."),
    )
    if extraction:
        doc.extraction = extraction
        await doc.asave()

    return DocumentOut(
        id=str(doc.id),
        file=doc.file.url if doc.file else "",
        content=doc.content,
        extraction=doc.extraction,
        file_name=doc.file_name or "",
        file_size=doc.file_size or 0,
        content_type=doc.content_type or "",
        file_extension=doc.file_extension or "",
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat(),
    )


@router.get("/{memory_id}/documents", response=list[DocumentOut])
def list_documents(request: HttpRequest, memory_id: str) -> list[DocumentOut]:
    """List all documents in a memory."""
    memory = Memory.objects.get(id=memory_id)
    docs = memory.documents.all().order_by("-created_at")
    return [
        DocumentOut(
            id=str(doc.id),
            file=doc.file.url if doc.file else "",
            content=doc.content,
            extraction=doc.extraction,
            file_name=doc.file_name or "",
            file_size=doc.file_size or 0,
            content_type=doc.content_type or "",
            file_extension=doc.file_extension or "",
            created_at=doc.created_at.isoformat(),
            updated_at=doc.updated_at.isoformat(),
        )
        for doc in docs
    ]


@router.get("/{memory_id}/documents/{doc_id}", response=DocumentOut)
def get_document(request: HttpRequest, memory_id: str, doc_id: str) -> DocumentOut:
    """Get a single document from a memory."""
    doc = Document.objects.get(id=doc_id, memory_id=memory_id)
    return DocumentOut(
        id=str(doc.id),
        file=doc.file.url if doc.file else "",
        content=doc.content,
        extraction=doc.extraction,
        file_name=doc.file_name or "",
        file_size=doc.file_size or 0,
        content_type=doc.content_type or "",
        file_extension=doc.file_extension or "",
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat(),
    )


@router.delete("/{memory_id}/documents/{doc_id}", response={204: None})
async def delete_document(
    request: HttpRequest, memory_id: str, doc_id: str
) -> tuple[int, None]:
    """Delete a document from a memory."""
    doc = await Document.objects.aget(id=doc_id, memory_id=memory_id)
    await doc.adelete()
    return 204, None


@router.post("/{memory_id}/link/{thread_id}", response={204: None})
async def link_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None]:
    """Link a memory to a thread."""
    memory = await Memory.objects.aget(id=memory_id)
    from django_ai_sdk.conversation.models import Thread

    thread = await Thread.objects.aget(id=thread_id)
    await ThreadMemory.objects.aget_or_create(
        thread=thread,
        memory=memory,
    )
    return 204, None


@router.delete("/{memory_id}/link/{thread_id}", response={204: None})
async def unlink_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None]:
    """Unlink a memory from a thread."""
    link = await ThreadMemory.objects.aget(memory_id=memory_id, thread_id=thread_id)
    await link.adelete()
    return 204, None


# ============================================================================
# Thread-Memory Management Endpoints
# ============================================================================


@router.get("/thread/{thread_id}", response=list[ThreadMemoryOut])
async def list_thread_memories(
    request: HttpRequest, thread_id: str
) -> list[ThreadMemoryOut]:
    """List all memories connected to a thread with their active status."""
    thread_memories_query = (
        ThreadMemory.objects.filter(thread_id=thread_id)
        .select_related("memory")
        .annotate(document_count=Count("memory__documents"))
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
    from django_ai_sdk.conversation.models import Thread

    thread = await Thread.objects.aget(id=thread_id)

    # Get or create links for all memories
    for memory_id in payload.memory_ids:
        memory = await Memory.objects.aget(id=memory_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
            defaults={"active": True},
        )

    # Return updated list
    return await list_thread_memories(request, thread_id)


@router.patch("/thread/{thread_id}/{memory_id}", response=ThreadMemoryOut)
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> ThreadMemoryOut:
    """Toggle the active status of a memory for a thread."""
    thread_memory = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
    thread_memory.active = payload.active
    await thread_memory.asave()

    # Fetch memory separately using async ORM
    memory = await Memory.objects.aget(id=memory_id)
    doc_count = await Document.objects.filter(memory_id=memory_id).acount()

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
