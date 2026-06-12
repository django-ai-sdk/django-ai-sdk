from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from django.conf import settings
from django.db.models import Count
from django.utils.module_loading import import_string

from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.files.services import FileService
from django_ai_sdk.memories.models import Entry, EntryDocument, Memory, MemoryOwner, ThreadMemory
from django_ai_sdk.memories.schemas import (
    DocumentOut,
    MemoryOut,
    MemoryOwnerOut,
    ThreadMemoryOut,
)
from django_ai_sdk.permissions import (
    BasePermission,
    Operation,
    check_object_permissions,
    check_permissions,
    get_default_permissions,
)

if TYPE_CHECKING:
    from typing import Any

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.core.files.base import File


@lru_cache(maxsize=1)
def _get_memory_permissions() -> list[type[BasePermission]]:
    """Resolve permission classes from settings (cached)."""
    paths = getattr(settings, "AI_SDK_MEMORY_PERMISSIONS", [])
    if not paths:
        return get_default_permissions()
    return [import_string(p) for p in paths]


async def _check_permission(
    user: AbstractBaseUser | AnonymousUser | None, operation: Operation
) -> None:
    """Permission check for memory operations."""
    await check_permissions(user, operation, _get_memory_permissions())


async def _check_object_permission(
    user: AbstractBaseUser | AnonymousUser | None, operation: Operation, obj: Any
) -> None:
    """Object permission check for memory operations."""
    permissions = _get_memory_permissions()
    await check_permissions(user, operation, permissions)
    await check_object_permissions(user, operation, obj, permissions)


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
    async def create_memory(
        name: str,
        description: str = "",
        slug: str = "",
        is_public: bool = True,
        *,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> MemoryOut:
        """Create a new memory."""
        await _check_permission(user, Operation.CREATE_MEMORY)

        memory = await Memory.objects.acreate(
            name=name,
            slug=slug,
            description=description,
            is_public=is_public,
        )
        if user and user.is_authenticated:
            await MemoryOwner.objects.acreate(
                memory=memory,
                user=user,
                can_manage=True,
            )
        return MemoryOut(
            id=str(memory.id),
            name=memory.name,
            slug=memory.slug,
            description=memory.description,
            is_public=memory.is_public,
            document_count=0,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    # ============================================================================
    # Memory Owner Management
    # ============================================================================

    @staticmethod
    async def list_owners(
        memory_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> list[MemoryOwnerOut]:
        """List all owners of a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.VIEW_MEMORY, memory)
        return [
            MemoryOwnerOut(
                user_id=str(o.user_id),
                can_manage=o.can_manage,
                created_at=o.created_at.isoformat(),
            )
            async for o in memory.owners.all().select_related("user")
        ]

    @staticmethod
    async def add_owner(
        memory_id: str,
        user_id: str,
        can_manage: bool = False,
        *,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> MemoryOwnerOut:
        """Add a user as owner of a memory."""
        from django.contrib.auth import get_user_model

        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UPDATE_MEMORY, memory)
        UserModel = get_user_model()
        owner_user = await UserModel.objects.aget(id=user_id)
        ownership, _created = await MemoryOwner.objects.aupdate_or_create(
            memory=memory,
            user=owner_user,
            defaults={"can_manage": can_manage},
        )
        return MemoryOwnerOut(
            user_id=str(ownership.user_id),
            can_manage=ownership.can_manage,
            created_at=ownership.created_at.isoformat(),
        )

    @staticmethod
    async def update_owner(
        memory_id: str,
        user_id: str,
        can_manage: bool,
        *,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> MemoryOwnerOut:
        """Update an owner's can_manage flag."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UPDATE_MEMORY, memory)
        ownership = await memory.owners.select_related("user").aget(user_id=user_id)
        ownership.can_manage = can_manage
        await ownership.asave(update_fields=["can_manage"])
        return MemoryOwnerOut(
            user_id=str(ownership.user_id),
            can_manage=ownership.can_manage,
            created_at=ownership.created_at.isoformat(),
        )

    @staticmethod
    async def remove_owner(
        memory_id: str, user_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Remove an owner from a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UPDATE_MEMORY, memory)
        ownership = await memory.owners.aget(user_id=user_id)
        await ownership.adelete()

    # ============================================================================

    @staticmethod
    async def list_memories(
        *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[MemoryOut]:
        """List all visible memories."""
        await _check_permission(user, Operation.VIEW_MEMORY)

        memories = (
            Memory.objects.filter(is_hidden=False)
            .annotate(document_count=Count("entries"))
            .order_by("-created_at")
        )
        return [
            MemoryOut(
                id=str(memory.id),
                name=memory.name,
                slug=memory.slug,
                description=memory.description,
                is_public=memory.is_public,
                document_count=memory.document_count,
                created_at=memory.created_at.isoformat(),
                updated_at=memory.updated_at.isoformat(),
            )
            async for memory in memories
        ]

    @staticmethod
    async def get_assistant_memories(assistant_id: str) -> list[str]:
        assistant = AssistantService.from_registry(assistant_id)
        return [
            str(m.id) async for m in Memory.objects.filter(slug__in=assistant.memories).distinct()
        ]

    @staticmethod
    async def link_memories(
        assistant_id: str, thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        for memory_id in await MemoryService.get_assistant_memories(assistant_id):
            await MemoryService.link_memory_to_thread(memory_id, thread_id, user=user)

    @staticmethod
    async def unlink_memories(
        assistant_id: str, thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        for memory_id in await MemoryService.get_assistant_memories(assistant_id):
            await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=user)

    @staticmethod
    async def get_memory(
        memory_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> MemoryOut:
        """Get a single memory by ID."""
        memory = await Memory.objects.annotate(document_count=Count("entries")).aget(id=memory_id)
        await _check_object_permission(user, Operation.VIEW_MEMORY, memory)
        return MemoryOut(
            id=str(memory.id),
            name=memory.name,
            slug=memory.slug,
            description=memory.description,
            is_public=memory.is_public,
            document_count=memory.document_count,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @staticmethod
    async def update_memory(
        memory_id: str,
        name: str,
        description: str = "",
        is_public: bool | None = None,
        *,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> MemoryOut:
        """Update a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UPDATE_MEMORY, memory)
        memory.name = name
        memory.description = description
        if is_public is not None:
            memory.is_public = is_public
        await memory.asave()
        doc_count = await memory.entries.acount()
        return MemoryOut(
            id=str(memory.id),
            name=memory.name,
            slug=memory.slug,
            description=memory.description,
            is_public=memory.is_public,
            document_count=doc_count,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @staticmethod
    async def delete_memory(
        memory_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Delete a memory and all its entries."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.DELETE_MEMORY, memory)
        await memory.adelete()

    # ============================================================================
    # Document CRUD (per memory)
    # ============================================================================

    @staticmethod
    async def upload_document(
        memory_id: str, file: File, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> DocumentOut | tuple[int, dict]:
        """Upload a file to a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UPLOAD_DOCUMENT, memory)

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)

        if result is None:
            return 400, {"detail": f"Unsupported or empty file: {file_name}"}

        entry = await Entry.objects.acreate(
            memory=memory,
            name=file_name,
        )

        entry_doc = await EntryDocument.objects.acreate(
            entry=entry,
            file=file,
            file_name=file_name,
            file_size=file.size or 0,
            content_type=getattr(file, "content_type", "") or "",
            file_extension=ext.lstrip("."),
        )

        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def list_documents(
        memory_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[DocumentOut]:
        """List all file-backed documents in a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.LIST_DOCUMENTS, memory)
        entry_docs = (
            EntryDocument.objects.filter(entry__memory_id=memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [MemoryService._entry_doc_to_out(ed) async for ed in entry_docs]

    @staticmethod
    async def get_document(
        memory_id: str, doc_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> DocumentOut:
        """Get a single document from a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.VIEW_DOCUMENT, memory)
        entry_doc = await EntryDocument.objects.select_related("entry").aget(
            entry_id=doc_id, entry__memory_id=memory_id
        )
        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def delete_document(
        memory_id: str, doc_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Delete a document (and its entry) from a memory."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.DELETE_DOCUMENT, memory)
        entry = await Entry.objects.aget(id=doc_id, memory_id=memory_id)
        await entry.adelete()

    # ============================================================================
    # Thread-Memory linking
    # ============================================================================

    @staticmethod
    async def link_memory_to_thread(
        memory_id: str, thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Link a memory to a thread."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.LINK_MEMORY, memory)
        thread = await Thread.objects.aget(id=thread_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
        )

    @staticmethod
    async def unlink_memory_from_thread(
        memory_id: str, thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Unlink a memory from a thread."""
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UNLINK_MEMORY, memory)
        link = await ThreadMemory.objects.aget(memory_id=memory_id, thread_id=thread_id)
        await link.adelete()

    @staticmethod
    async def list_thread_memories(
        thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[ThreadMemoryOut]:
        """List all memories connected to a thread with their active status."""
        thread_memories_query = (
            ThreadMemory.objects.filter(thread_id=thread_id, memory__is_hidden=False)
            .select_related("memory")
            .annotate(document_count=Count("memory__entries"))
        )

        memories = []
        async for tm in thread_memories_query:
            await _check_object_permission(user, Operation.LIST_THREAD_MEMORIES, tm.memory)
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
    async def bulk_connect_memories(
        thread_id: str, memory_ids: list[str], *, user: AbstractBaseUser | AnonymousUser | None
    ) -> list[ThreadMemoryOut]:
        """Connect multiple memories to a thread at once."""
        thread = await Thread.objects.aget(id=thread_id)

        memories = {str(m.id): m async for m in Memory.objects.filter(id__in=memory_ids)}

        links = []
        for memory_id in memory_ids:
            memory = memories.get(memory_id)
            if not memory:
                continue
            await _check_object_permission(user, Operation.LINK_MEMORY, memory)
            links.append(ThreadMemory(thread=thread, memory=memory, active=True))

        await ThreadMemory.objects.abulk_create(links, ignore_conflicts=True)

        return await MemoryService.list_thread_memories(thread_id, user=user)

    @staticmethod
    async def toggle_memory_active(
        thread_id: str,
        memory_id: str,
        active: bool,
        *,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> ThreadMemoryOut:
        """Toggle the active status of a memory for a thread."""
        thread_memory = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.LINK_MEMORY, memory)
        thread_memory.active = active
        await thread_memory.asave()

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
    async def disconnect_memory_from_thread(
        thread_id: str, memory_id: str, *, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Disconnect a memory from a thread."""
        link = await ThreadMemory.objects.aget(thread_id=thread_id, memory_id=memory_id)
        memory = await Memory.objects.aget(id=memory_id)
        await _check_object_permission(user, Operation.UNLINK_MEMORY, memory)
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
    async def upload_thread_file(
        thread_id: str, file: File, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> DocumentOut | tuple[int, dict]:
        """Upload a file to a thread. Auto-creates a hidden memory on first upload."""
        thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
        await _check_object_permission(user, Operation.UPLOAD_FILE, thread)
        memory = await MemoryService.get_or_create_thread_file_memory(thread_id)

        # get assistant from thread
        assistant = await AssistantService.get_assistant(thread_id)

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)

        if result is None:
            return 400, {"detail": f"Unsupported or empty file: {file_name}"}

        entry = await Entry.objects.acreate(
            memory=memory,
            name=file_name,
        )

        entry_doc = await EntryDocument.objects.acreate(
            entry=entry,
            file=file,
            file_name=file_name,
            file_size=file.size or 0,
            content_type=getattr(file, "content_type", "") or "",
            file_extension=ext.lstrip("."),
        )

        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def list_thread_files(
        thread_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[DocumentOut]:
        """List all files uploaded to a thread."""
        try:
            thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
        except Thread.DoesNotExist:
            return []

        await _check_object_permission(user, Operation.VIEW_FILE, thread)

        if not thread.file_memory_id:
            return []

        entry_docs = (
            EntryDocument.objects.filter(entry__memory_id=thread.file_memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [MemoryService._entry_doc_to_out(ed) async for ed in entry_docs]

    @staticmethod
    async def delete_thread_file(
        thread_id: str, doc_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> None:
        """Delete a file from a thread."""
        thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
        await _check_object_permission(user, Operation.DELETE_FILE, thread)
        if not thread.file_memory_id:
            return
        entry = await Entry.objects.aget(id=doc_id, memory_id=thread.file_memory_id)
        await entry.adelete()

    @staticmethod
    async def get_chunk_content(
        entry_id: str, chunk_id: str | None, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> str | None:
        """Return a specific chunk from the vector store, falling back to full Entry content.

        When chunk_id is None the document was not split into chunks, so the
        full Entry content is returned directly without querying the vector store.

        Chunk lookup uses the already-warmed RAG cache (get_cached_rag_instance) to
        avoid opening a second connection on backends with exclusive file locks (Qdrant).
        Falls back to entry.content when no warmed RAG is available for this memory.
        """
        from django_ai_sdk.assistants.registry import registry

        try:
            entry = await Entry.objects.aget(id=entry_id)
        except Entry.DoesNotExist:
            return None

        await _check_object_permission(user, Operation.VIEW_DOCUMENT, entry)

        if chunk_id is None:
            return entry.content

        memory_id = str(entry.memory_id)

        for assistant in registry.all().values():
            if assistant.rag_provider is None:
                continue
            rag = assistant.rag_provider.get_cached_rag_instance(assistant, memory_id)
            if rag is None:
                continue
            chunk = await rag.get_chunk(chunk_id)
            if chunk is not None:
                return chunk

        return entry.content

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
link_memories = async_to_sync(MemoryService.link_memories)
unlink_memories = async_to_sync(MemoryService.unlink_memories)
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
get_chunk_content = async_to_sync(MemoryService.get_chunk_content)
list_owners = async_to_sync(MemoryService.list_owners)
add_owner = async_to_sync(MemoryService.add_owner)
update_owner = async_to_sync(MemoryService.update_owner)
remove_owner = async_to_sync(MemoryService.remove_owner)
