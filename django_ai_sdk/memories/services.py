from typing import Any

from asgiref.sync import async_to_sync
from django.db.models import Count

from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, EntryDocument, Memory, ThreadMemory
from django_ai_sdk.memories.schemas import (
    DocumentOut,
    MemoryOut,
    ThreadMemoryOut,
)
from django_ai_sdk.memories.utils import extract_document, read_text_content


class MemoryService:
    """
    Service for memory operations.

    All methods are async. Use the sync-prefixed aliases for sync contexts
    (e.g., DRF class-based views).
    """

    # ============================================================================
    # Memory CRUD
    # ============================================================================

    @staticmethod
    async def create_memory(name: str, description: str = "") -> MemoryOut:
        """Create a new memory."""
        memory = await Memory.objects.acreate(
            name=name,
            description=description,
        )
        return MemoryOut(
            id=str(memory.id),
            name=memory.name,
            description=memory.description,
            document_count=0,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @staticmethod
    async def list_memories() -> list[MemoryOut]:
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
            async for memory in memories
        ]

    @staticmethod
    async def get_memory(memory_id: str) -> MemoryOut:
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

    @staticmethod
    async def update_memory(memory_id: str, name: str, description: str = "") -> MemoryOut:
        """Update a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        memory.name = name
        memory.description = description
        await memory.asave()
        doc_count = await memory.entries.acount()
        return MemoryOut(
            id=str(memory.id),
            name=memory.name,
            description=memory.description,
            document_count=doc_count,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @staticmethod
    async def delete_memory(memory_id: str) -> None:
        """Delete a memory and all its entries."""
        memory = await Memory.objects.aget(id=memory_id)
        await memory.adelete()

    # ============================================================================
    # Document CRUD (per memory)
    # ============================================================================

    @staticmethod
    async def upload_document(memory_id: str, file: Any) -> DocumentOut | tuple[int, dict]:
        """Upload a file to a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        file_name = file.name or ""

        result = read_text_content(file)
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

        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def list_documents(memory_id: str) -> list[DocumentOut]:
        """List all file-backed documents in a memory."""
        entry_docs = (
            EntryDocument.objects.filter(entry__memory_id=memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [MemoryService._entry_doc_to_out(ed) for ed in entry_docs]

    @staticmethod
    async def get_document(memory_id: str, doc_id: str) -> DocumentOut:
        """Get a single document from a memory."""
        entry_doc = await EntryDocument.objects.select_related("entry").aget(
            entry_id=doc_id, entry__memory_id=memory_id
        )
        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def delete_document(memory_id: str, doc_id: str) -> None:
        """Delete a document (and its entry) from a memory."""
        entry = await Entry.objects.aget(id=doc_id, memory_id=memory_id)
        await entry.adelete()

    # ============================================================================
    # Thread-Memory linking
    # ============================================================================

    @staticmethod
    async def link_memory_to_thread(memory_id: str, thread_id: str) -> None:
        """Link a memory to a thread."""
        memory = await Memory.objects.aget(id=memory_id)
        thread = await Thread.objects.aget(id=thread_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
        )

    @staticmethod
    async def unlink_memory_from_thread(memory_id: str, thread_id: str) -> None:
        """Unlink a memory from a thread."""
        link = await ThreadMemory.objects.aget(memory_id=memory_id, thread_id=thread_id)
        await link.adelete()

    @staticmethod
    async def list_thread_memories(thread_id: str) -> list[ThreadMemoryOut]:
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

    @staticmethod
    async def bulk_connect_memories(thread_id: str, memory_ids: list[str]) -> list[ThreadMemoryOut]:
        """Connect multiple memories to a thread at once."""
        thread = await Thread.objects.aget(id=thread_id)

        for memory_id in memory_ids:
            memory = await Memory.objects.aget(id=memory_id)
            await ThreadMemory.objects.aget_or_create(
                thread=thread,
                memory=memory,
                defaults={"active": True},
            )

        return await MemoryService.list_thread_memories(thread_id)

    @staticmethod
    async def toggle_memory_active(thread_id: str, memory_id: str, active: bool) -> ThreadMemoryOut:
        """Toggle the active status of a memory for a thread."""
        thread_memory = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
        thread_memory.active = active
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

    @staticmethod
    async def disconnect_memory_from_thread(thread_id: str, memory_id: str) -> None:
        """Disconnect a memory from a thread."""
        link = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
        await link.adelete()

    # ============================================================================
    # Thread file uploads
    # ============================================================================

    @staticmethod
    async def get_or_create_thread_file_memory(thread_id: str) -> Memory:
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

    @staticmethod
    async def upload_thread_file(thread_id: str, file: Any) -> DocumentOut | tuple[int, dict]:
        """Upload a file to a thread. Auto-creates a hidden memory on first upload."""
        memory = await MemoryService.get_or_create_thread_file_memory(thread_id)
        file_name = file.name or ""

        result = read_text_content(file)
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

        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def list_thread_files(thread_id: str) -> list[DocumentOut]:
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
        return [MemoryService._entry_doc_to_out(ed) for ed in entry_docs]

    @staticmethod
    async def delete_thread_file(thread_id: str, doc_id: str) -> None:
        """Delete a file from a thread."""
        thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
        if not thread.file_memory_id:
            return
        entry = await Entry.objects.aget(id=doc_id, memory_id=thread.file_memory_id)
        await entry.adelete()

    # ============================================================================
    # Private helpers
    # ============================================================================

    @staticmethod
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


# ============================================================================
# Sync wrappers for use in sync contexts
# ============================================================================

create_memory = async_to_sync(MemoryService.create_memory)
list_memories = async_to_sync(MemoryService.list_memories)
get_memory = async_to_sync(MemoryService.get_memory)
update_memory = async_to_sync(MemoryService.update_memory)
delete_memory = async_to_sync(MemoryService.delete_memory)
upload_document = async_to_sync(MemoryService.upload_document)
list_documents = async_to_sync(MemoryService.list_documents)
get_document = async_to_sync(MemoryService.get_document)
delete_document = async_to_sync(MemoryService.delete_document)
link_memory_to_thread = async_to_sync(MemoryService.link_memory_to_thread)
unlink_memory_from_thread = async_to_sync(MemoryService.unlink_memory_from_thread)
list_thread_memories = async_to_sync(MemoryService.list_thread_memories)
bulk_connect_memories = async_to_sync(MemoryService.bulk_connect_memories)
toggle_memory_active = async_to_sync(MemoryService.toggle_memory_active)
disconnect_memory_from_thread = async_to_sync(MemoryService.disconnect_memory_from_thread)
get_or_create_thread_file_memory = async_to_sync(MemoryService.get_or_create_thread_file_memory)
upload_thread_file = async_to_sync(MemoryService.upload_thread_file)
list_thread_files = async_to_sync(MemoryService.list_thread_files)
delete_thread_file = async_to_sync(MemoryService.delete_thread_file)
