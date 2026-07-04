from __future__ import annotations

import os
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.db.models import Count

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import (
    Entry,
    EntryDocument,
    Memory,
    MemoryGroup,
    MemoryUser,
    ThreadMemory,
)
from django_ai_sdk.memories.schemas import (
    DocumentOut,
    DocumentStatusOut,
    DocumentUploadResponse,
    MemoryGroupOut,
    MemoryOut,
    MemoryUserOut,
    ThreadMemoryOut,
)
from django_ai_sdk.memories.tasks import process_document_upload
from django_ai_sdk.permissions import (
    Operation,
    get_assistant_permissions,
    get_memory_permissions,
    get_queryset_perms,
    has_perms,
)
from django_ai_sdk.tasks import aget_task_status

if TYPE_CHECKING:
    from typing import Any

    from django.core.files.base import File

    from django_ai_sdk.types import UserType


async def _aget_or_not_found(qs: Any, **lookup: Any) -> Any:
    """Fetch a single object or raise ValueError (mapped to 404 at the view layer)."""
    from django.core.exceptions import ObjectDoesNotExist

    try:
        return await qs.aget(**lookup)
    except ObjectDoesNotExist:
        raise ValueError(f"{qs.model.__name__} not found") from None


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
        user: UserType,
    ) -> MemoryOut:
        """Create a new memory."""
        await has_perms(user, Operation.CREATE_MEMORY, permissions=get_memory_permissions())

        memory = await Memory.objects.acreate(
            name=name,
            slug=slug,
            description=description,
            is_public=is_public,
        )
        if user and user.is_authenticated:
            await MemoryUser.objects.acreate(
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
    # Memory User Management
    # ============================================================================

    @staticmethod
    async def list_memory_users(memory_id: str, *, user: UserType) -> list[MemoryUserOut]:
        """List all users of a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.VIEW_MEMORY, memory, permissions=get_memory_permissions())
        return [
            MemoryUserOut(
                user_id=str(o.user_id),
                can_manage=o.can_manage,
                created_at=o.created_at.isoformat(),
            )
            async for o in memory.memory_users.all().select_related("user")
        ]

    @staticmethod
    async def add_memory_user(
        memory_id: str,
        user_id: str,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> MemoryUserOut:
        """Add a user to a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
        UserModel = get_user_model()
        target_user = await _aget_or_not_found(UserModel.objects, id=user_id)
        ownership, _created = await MemoryUser.objects.aupdate_or_create(
            memory=memory,
            user=target_user,
            defaults={"can_manage": can_manage},
        )
        return MemoryUserOut(
            user_id=str(ownership.user_id),
            can_manage=ownership.can_manage,
            created_at=ownership.created_at.isoformat(),
        )

    @staticmethod
    async def update_memory_user(
        memory_id: str,
        user_id: str,
        can_manage: bool,
        *,
        user: UserType,
    ) -> MemoryUserOut:
        """Update a memory user's can_manage flag."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
        ownership = await _aget_or_not_found(
            memory.memory_users.select_related("user"), user_id=user_id
        )
        ownership.can_manage = can_manage
        await ownership.asave(update_fields=["can_manage"])
        return MemoryUserOut(
            user_id=str(ownership.user_id),
            can_manage=ownership.can_manage,
            created_at=ownership.created_at.isoformat(),
        )

    @staticmethod
    async def remove_memory_user(memory_id: str, user_id: str, *, user: UserType) -> None:
        """Remove a user from a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
        ownership = await _aget_or_not_found(memory.memory_users, user_id=user_id)
        await ownership.adelete()

    @staticmethod
    async def add_memory_group(
        memory_id: str,
        group_id: int,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> MemoryGroupOut:
        from django.contrib.auth.models import Group

        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
        group = await _aget_or_not_found(Group.objects, id=group_id)
        obj, _created = await MemoryGroup.objects.aupdate_or_create(
            memory=memory,
            group=group,
            defaults={"can_manage": can_manage},
        )
        return MemoryGroupOut(
            group_id=obj.group_id,
            group_name=group.name,
            can_manage=obj.can_manage,
            created_at=obj.created_at.isoformat(),
        )

    @staticmethod
    async def remove_memory_group(
        memory_id: str,
        group_id: int,
        *,
        user: UserType,
    ) -> None:
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
        await memory.memory_groups.filter(group_id=group_id).adelete()

    @staticmethod
    async def list_memory_groups(
        memory_id: str,
        *,
        user: UserType,
    ) -> list[MemoryGroupOut]:
        """List all groups of a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.VIEW_MEMORY, memory, permissions=get_memory_permissions())
        return [
            MemoryGroupOut(
                group_id=mg.group_id,
                group_name=mg.group.name,
                can_manage=mg.can_manage,
                created_at=mg.created_at.isoformat(),
            )
            async for mg in memory.memory_groups.all().select_related("group")
        ]

    # ============================================================================

    @staticmethod
    async def list_memories(*, user: UserType) -> list[MemoryOut]:
        """List memories visible to the requesting user."""
        qs = get_queryset_perms(
            user,
            Operation.VIEW_MEMORY,
            queryset=Memory.objects.filter(is_hidden=False),
            permissions=get_memory_permissions(),
        )
        qs = qs.annotate(document_count=Count("entries")).order_by("-created_at")

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
            async for memory in qs
        ]

    @staticmethod
    async def _get_assistant_memory_objects(assistant_id: str) -> list[Memory]:
        """Resolve the assistant's configured memories in a single query."""
        try:
            assistant = await AssistantService.get(assistant_id)
        except ValueError:
            return []
        slugs: list[str] = getattr(assistant, "memories", [])
        if not slugs:
            return []
        return [m async for m in Memory.objects.filter(slug__in=slugs).distinct()]

    @staticmethod
    async def get_assistant_memories(assistant_id: str) -> list[str]:
        memories = await MemoryService._get_assistant_memory_objects(assistant_id)
        return [str(m.id) for m in memories]

    @staticmethod
    async def _filter_memories_by_perm(
        memories: list[Memory],
        operation: Operation,
        user: UserType,
    ) -> list[Memory]:
        """Keep only memories the user has *operation* permission for (silent skip)."""
        result: list[Memory] = []
        for memory in memories:
            if await has_perms(
                user,
                operation,
                memory,
                permissions=get_memory_permissions(),
                raise_on_deny=False,
            ):
                result.append(memory)
        return result

    @staticmethod
    async def get_assistant_memories_for_user(assistant_id: str, *, user: UserType) -> list[Memory]:
        """Return the assistant's configured memories the user can ``VIEW_MEMORY``.

        Filters out memories the user doesn't have permission to see, so the
        caller never receives unreadable memories.
        """
        memories = await MemoryService._get_assistant_memory_objects(assistant_id)
        return await MemoryService._filter_memories_by_perm(memories, Operation.VIEW_MEMORY, user)

    @staticmethod
    async def link_memories(
        assistant_id: str,
        thread_id: str,
        *,
        user: UserType,
    ) -> None:
        """Link an assistant's configured memories to a thread.

        Skips memories the user doesn't have ``LINK_MEMORY`` permission for,
        consistent with the per-request retrieval filter, a private memory
        the user cannot read is simply not linked, rather than blocking by raise.
        """
        memories = await MemoryService._get_assistant_memory_objects(assistant_id)
        for memory in await MemoryService._filter_memories_by_perm(
            memories, Operation.LINK_MEMORY, user
        ):
            await ThreadMemory.objects.aget_or_create(thread_id=thread_id, memory=memory)

    @staticmethod
    async def unlink_memories(
        assistant_id: str,
        thread_id: str,
        *,
        user: UserType,
    ) -> None:
        """Unlink an assistant's configured memories from a thread.

        Only unlinks memories the user has ``UNLINK_MEMORY`` permission for.
        """
        memories = await MemoryService._get_assistant_memory_objects(assistant_id)
        allowed = await MemoryService._filter_memories_by_perm(
            memories, Operation.UNLINK_MEMORY, user
        )
        if not allowed:
            return
        await ThreadMemory.objects.filter(
            thread_id=thread_id,
            memory_id__in=[m.id for m in allowed],
        ).adelete()

    @staticmethod
    async def get_memory(memory_id: str, *, user: UserType) -> MemoryOut:
        """Get a single memory by ID."""
        memory = await _aget_or_not_found(
            Memory.objects.annotate(document_count=Count("entries")), id=memory_id
        )
        await has_perms(user, Operation.VIEW_MEMORY, memory, permissions=get_memory_permissions())
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
        user: UserType,
    ) -> MemoryOut:
        """Update a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UPDATE_MEMORY, memory, permissions=get_memory_permissions())
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
    async def delete_memory(memory_id: str, *, user: UserType) -> None:
        """Delete a memory and all its entries."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.DELETE_MEMORY, memory, permissions=get_memory_permissions())
        await memory.adelete()

    # ============================================================================
    # Document CRUD (per memory)
    # ============================================================================

    @staticmethod
    async def upload_document(
        memory_id: str, file: File, *, user: UserType
    ) -> DocumentUploadResponse:
        """Save file and enqueue pipeline processing. Returns immediately with doc_id."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(
            user, Operation.UPLOAD_DOCUMENT, memory, permissions=get_memory_permissions()
        )

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)

        entry_doc = await EntryDocument.objects.acreate(
            entry=None,
            memory=memory,
            file=file,
            file_name=file_name,
            file_size=file.size or 0,
            content_type=getattr(file, "content_type", "") or "",
            file_extension=ext.lstrip("."),
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        )

        task_result = await process_document_upload.aenqueue(str(entry_doc.id), memory_id)
        await EntryDocument.objects.filter(
            id=entry_doc.id,
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        ).aupdate(
            task_id=task_result.id,
            processing_status=EntryDocument.ProcessingStatus.PROCESSING,
        )

        return DocumentUploadResponse(id=str(entry_doc.id), status="processing")

    @staticmethod
    async def list_documents(memory_id: str, *, user: UserType) -> list[DocumentOut]:
        """List all file-backed documents in a memory (all processing statuses)."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(
            user, Operation.LIST_DOCUMENTS, memory, permissions=get_memory_permissions()
        )
        entry_docs = (
            EntryDocument.objects.filter(memory_id=memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [MemoryService._entry_doc_to_out(ed) async for ed in entry_docs]

    @staticmethod
    async def get_document(memory_id: str, doc_id: str, *, user: UserType) -> DocumentOut:
        """Get a single document from a memory by its EntryDocument id."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.VIEW_DOCUMENT, memory, permissions=get_memory_permissions())
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("entry"),
            id=doc_id,
            memory_id=memory_id,
        )
        return MemoryService._entry_doc_to_out(entry_doc)

    @staticmethod
    async def delete_document(memory_id: str, doc_id: str, *, user: UserType) -> None:
        """Delete a document from a memory by its EntryDocument id.

        Works for in-flight docs (no Entry yet). When an Entry exists, deleting it
        cascades to the EntryDocument and triggers RAG cleanup via the delete signal.
        """
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(
            user, Operation.DELETE_DOCUMENT, memory, permissions=get_memory_permissions()
        )
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("entry"),
            id=doc_id,
            memory_id=memory_id,
        )
        if entry_doc.entry_id:
            await entry_doc.entry.adelete()
        else:
            await entry_doc.adelete()

    # ============================================================================
    # Thread-Memory linking
    # ============================================================================

    @staticmethod
    async def link_memory_to_thread(memory_id: str, thread_id: str, *, user: UserType) -> None:
        """Link a memory to a thread."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.LINK_MEMORY, memory, permissions=get_memory_permissions())
        thread = await _aget_or_not_found(Thread.objects, id=thread_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
        )

    @staticmethod
    async def unlink_memory_from_thread(memory_id: str, thread_id: str, *, user: UserType) -> None:
        """Unlink a memory from a thread."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UNLINK_MEMORY, memory, permissions=get_memory_permissions())
        link = await _aget_or_not_found(
            ThreadMemory.objects, memory_id=memory_id, thread_id=thread_id
        )
        await link.adelete()

    @staticmethod
    async def list_thread_memories(thread_id: str, *, user: UserType) -> list[ThreadMemoryOut]:
        """List all memories connected to a thread with their active status."""
        thread_memories = (
            ThreadMemory.objects.filter(thread_id=thread_id, memory__is_hidden=False)
            .select_related("memory")
            .annotate(document_count=Count("memory__entries"))
        )

        memories = []
        async for tm in thread_memories:
            # Skip memories this user cannot read instead of raising: a thread may
            # carry assistant-linked private memories the user has no access to.
            if not await has_perms(
                user,
                Operation.VIEW_MEMORY,
                tm.memory,
                permissions=get_memory_permissions(),
                raise_on_deny=False,
            ):
                continue
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
    async def get_thread_memories(thread_id: str, *, user: UserType) -> list[Memory]:
        """Return active Memory objects for a thread that ``user`` can read.

        Filters out memories the user lacks ``VIEW_MEMORY`` access to (silent
        skip), so the caller never sees unreadable memories.
        """
        query = (
            ThreadMemory.objects.filter(thread_id=thread_id, active=True)
            .select_related("memory")
            .prefetch_related("memory__memory_users")
        )
        result: list[Memory] = []
        async for tm in query:
            if tm.memory is None:
                continue
            if not await has_perms(
                user,
                Operation.VIEW_MEMORY,
                tm.memory,
                permissions=get_memory_permissions(),
                raise_on_deny=False,
            ):
                continue
            result.append(tm.memory)
        return result

    @staticmethod
    async def bulk_connect_memories(
        thread_id: str, memory_ids: list[str], *, user: UserType
    ) -> list[ThreadMemoryOut]:
        """Connect multiple memories to a thread at once.

        Raises:
            ValueError: If any of the requested memory ids does not exist.
            PermissionDenied: If the user lacks LINK_MEMORY on any memory.
        """
        thread = await _aget_or_not_found(Thread.objects, id=thread_id)

        memories = {str(m.id): m async for m in Memory.objects.filter(id__in=memory_ids)}
        missing = [mid for mid in memory_ids if mid not in memories]
        if missing:
            raise ValueError(f"Memories not found: {', '.join(missing)}")

        links = []
        for memory_id in memory_ids:
            memory = memories[memory_id]
            await has_perms(
                user, Operation.LINK_MEMORY, memory, permissions=get_memory_permissions()
            )
            links.append(ThreadMemory(thread=thread, memory=memory, active=True))

        await ThreadMemory.objects.abulk_create(links, ignore_conflicts=True)

        return await MemoryService.list_thread_memories(thread_id, user=user)

    @staticmethod
    async def toggle_memory_active(
        thread_id: str,
        memory_id: str,
        active: bool,
        *,
        user: UserType,
    ) -> ThreadMemoryOut:
        """Toggle the active status of a memory for a thread."""
        thread_memory = await _aget_or_not_found(
            ThreadMemory.objects, thread_id=thread_id, memory_id=memory_id
        )
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.LINK_MEMORY, memory, permissions=get_memory_permissions())
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
        thread_id: str, memory_id: str, *, user: UserType
    ) -> None:
        """Disconnect a memory from a thread."""
        link = await _aget_or_not_found(
            ThreadMemory.objects, thread_id=thread_id, memory_id=memory_id
        )
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await has_perms(user, Operation.UNLINK_MEMORY, memory, permissions=get_memory_permissions())
        await link.adelete()

    # ============================================================================
    # Thread file uploads
    # ============================================================================

    @staticmethod
    async def get_or_create_thread_file_memory(thread_id: str) -> Memory:
        """Get or auto-create the hidden file-upload Memory for a thread."""
        thread = await _aget_or_not_found(
            Thread.objects.select_related("file_memory"), id=thread_id
        )
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
        thread_id: str, file: File, *, user: UserType
    ) -> DocumentUploadResponse:
        """Save file and enqueue pipeline processing. Returns immediately with doc_id."""
        thread = await _aget_or_not_found(
            Thread.objects.select_related("file_memory"), id=thread_id
        )
        assistant_id = thread.metadata.get("assistant_id") or None
        if assistant_id is None:
            raise ValueError("Thread has no assistant")
        assistant = await AssistantService.get(assistant_id)
        await has_perms(
            user,
            Operation.UPLOAD_FILE,
            thread,
            permissions=get_assistant_permissions(assistant),
        )
        memory = await MemoryService.get_or_create_thread_file_memory(thread_id)

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)

        entry_doc = await EntryDocument.objects.acreate(
            entry=None,
            memory=memory,
            file=file,
            file_name=file_name,
            file_size=file.size or 0,
            content_type=getattr(file, "content_type", "") or "",
            file_extension=ext.lstrip("."),
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        )

        task_result = await process_document_upload.aenqueue(
            str(entry_doc.id), str(memory.id), assistant_id
        )
        await EntryDocument.objects.filter(
            id=entry_doc.id,
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        ).aupdate(
            task_id=task_result.id,
            processing_status=EntryDocument.ProcessingStatus.PROCESSING,
        )

        return DocumentUploadResponse(id=str(entry_doc.id), status="processing")

    @staticmethod
    async def list_thread_files(thread_id: str, *, user: UserType) -> list[DocumentOut]:
        """List all files uploaded to a thread."""
        thread = await _aget_or_not_found(
            Thread.objects.select_related("file_memory"), id=thread_id
        )

        assistant_id = thread.metadata.get("assistant_id") or None
        if assistant_id is None:
            raise ValueError("Thread has no assistant")
        assistant = await AssistantService.get(assistant_id)
        await has_perms(
            user,
            Operation.VIEW_FILE,
            thread,
            permissions=get_assistant_permissions(assistant),
        )

        if not thread.file_memory_id:
            return []

        entry_docs = (
            EntryDocument.objects.filter(memory_id=thread.file_memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [MemoryService._entry_doc_to_out(ed) async for ed in entry_docs]

    @staticmethod
    async def delete_thread_file(thread_id: str, doc_id: str, *, user: UserType) -> None:
        """Delete a file from a thread by its EntryDocument id.

        Works for in-flight docs (no Entry yet), so a slow/stuck upload can be
        cancelled to unblock a deferred send.
        """
        thread = await _aget_or_not_found(
            Thread.objects.select_related("file_memory"), id=thread_id
        )
        assistant_id = thread.metadata.get("assistant_id") or None
        if assistant_id is None:
            raise ValueError("Thread has no assistant")
        assistant = await AssistantService.get(assistant_id)
        await has_perms(
            user,
            Operation.DELETE_FILE,
            thread,
            permissions=get_assistant_permissions(assistant),
        )
        if not thread.file_memory_id:
            return
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("entry"),
            id=doc_id,
            memory_id=thread.file_memory_id,
        )
        if entry_doc.entry_id:
            await entry_doc.entry.adelete()
        else:
            await entry_doc.adelete()

    @staticmethod
    async def get_document_status(doc_id: str, *, user: UserType) -> DocumentStatusOut:
        """Return processing status for a document.

        Requires ``VIEW_DOCUMENT`` on the document's owning memory.  Documents
        whose ``memory`` is ``None`` (legacy rows predating migration 0002) are
        accessible without a permission check to preserve backward compatibility
        — they carry no content, only a status string.
        """
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("memory"), id=doc_id
        )
        if entry_doc.memory_id is not None:
            await has_perms(
                user,
                Operation.VIEW_DOCUMENT,
                entry_doc.memory,
                permissions=get_memory_permissions(),
            )
        task_status = None
        if entry_doc.task_id:
            try:
                task_status = await aget_task_status(entry_doc.task_id)
            except Exception:
                pass
        return DocumentStatusOut(
            id=str(entry_doc.id),
            status=entry_doc.processing_status,
            error=entry_doc.processing_error,
            task=task_status,
        )

    @staticmethod
    async def retry_document(doc_id: str, *, user: UserType) -> DocumentStatusOut:
        """Re-enqueue processing for a failed or stuck document.

        Only documents in ``FAILED`` or ``PENDING`` status may be retried.
        ``PROCESSING`` documents are skipped (the worker is still active) and
        ``COMPLETED`` documents are skipped (re-processing would create duplicate
        index entries).

        Recovers ``assistant_id`` from the owning thread (for thread-file
        memories) so the assistant's file pipeline (e.g. OCR) is used again,
        not just the default text pipeline.
        """
        _RETRYABLE = {
            EntryDocument.ProcessingStatus.FAILED,
            EntryDocument.ProcessingStatus.PENDING,
        }

        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("memory"), id=doc_id
        )
        memory = entry_doc.memory
        if memory is None:
            raise ValueError("Document has no associated memory")

        if entry_doc.processing_status not in _RETRYABLE:
            raise ValueError(
                f"Document cannot be retried in status {entry_doc.processing_status!r}. "
                f"Only {sorted(s.value for s in _RETRYABLE)} are retryable."
            )

        assistant_id = None
        thread = await Thread.objects.filter(file_memory_id=memory.id).afirst()
        if thread is not None:
            assistant_id = thread.metadata.get("assistant_id")
            if assistant_id is None:
                raise ValueError("Thread has no assistant")
            assistant = await AssistantService.get(assistant_id)
            await has_perms(
                user,
                Operation.UPLOAD_FILE,
                thread,
                permissions=get_assistant_permissions(assistant),
            )
        else:
            await has_perms(
                user, Operation.UPLOAD_DOCUMENT, memory, permissions=get_memory_permissions()
            )

        entry_doc.processing_status = EntryDocument.ProcessingStatus.PENDING
        entry_doc.processing_error = ""
        await entry_doc.asave(update_fields=["processing_status", "processing_error", "updated_at"])

        task_result = await process_document_upload.aenqueue(
            str(entry_doc.id), str(memory.id), assistant_id
        )
        await EntryDocument.objects.filter(
            id=entry_doc.id,
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        ).aupdate(
            task_id=task_result.id,
            processing_status=EntryDocument.ProcessingStatus.PROCESSING,
        )
        # Re-read to return the actual persisted state — the conditional aupdate above
        # may have been skipped if the worker already advanced past PENDING.
        await entry_doc.arefresh_from_db(
            fields=["processing_status", "processing_error", "task_id"]
        )
        return DocumentStatusOut(
            id=str(entry_doc.id),
            status=entry_doc.processing_status,
            error=entry_doc.processing_error,
            task=None,
        )

    @staticmethod
    async def get_chunk_content(
        entry_id: str, chunk_id: str | None, *, user: UserType
    ) -> str | None:
        """Return a specific chunk from the vector store, falling back to full Entry content.

        When chunk_id is None the document was not split into chunks, so the
        full Entry content is returned directly without querying the vector store.

        Chunk lookup uses the already-warmed RAG cache (get_cached_rag_instance) to
        avoid opening a second connection on backends with exclusive file locks (Qdrant).
        Falls back to entry.content when no warmed RAG is available for this memory.
        """
        try:
            entry = await Entry.objects.select_related("memory").aget(id=entry_id)
        except Entry.DoesNotExist:
            return None

        await has_perms(
            user, Operation.VIEW_DOCUMENT, entry.memory, permissions=get_memory_permissions()
        )

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
        # `entry` is None until processing produces one (in-flight / failed docs).
        # The document id is always the EntryDocument id so it's stable across the
        # whole lifecycle.
        entry = entry_doc.entry
        return DocumentOut(
            id=str(entry_doc.id),
            file=entry_doc.file.url if entry_doc.file else "",
            content=entry.content if entry else "",
            extraction=entry.extraction if entry else None,
            file_name=entry_doc.file_name,
            data=(entry.data or {}) if entry else {},
            file_size=entry_doc.file_size,
            content_type=entry_doc.content_type,
            file_extension=entry_doc.file_extension,
            status=entry_doc.processing_status,
            error=entry_doc.processing_error,
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
get_thread_memories = async_to_sync(MemoryService.get_thread_memories)
bulk_connect_memories = async_to_sync(MemoryService.bulk_connect_memories)
toggle_memory_active = async_to_sync(MemoryService.toggle_memory_active)
disconnect_memory_from_thread = async_to_sync(MemoryService.disconnect_memory_from_thread)
get_or_create_thread_file_memory = async_to_sync(MemoryService.get_or_create_thread_file_memory)
upload_thread_file = async_to_sync(MemoryService.upload_thread_file)
list_thread_files = async_to_sync(MemoryService.list_thread_files)
delete_thread_file = async_to_sync(MemoryService.delete_thread_file)
get_document_status = async_to_sync(MemoryService.get_document_status)
retry_document = async_to_sync(MemoryService.retry_document)
get_chunk_content = async_to_sync(MemoryService.get_chunk_content)
list_memory_users = async_to_sync(MemoryService.list_memory_users)
add_memory_user = async_to_sync(MemoryService.add_memory_user)
update_memory_user = async_to_sync(MemoryService.update_memory_user)
remove_memory_user = async_to_sync(MemoryService.remove_memory_user)
add_memory_group = async_to_sync(MemoryService.add_memory_group)
remove_memory_group = async_to_sync(MemoryService.remove_memory_group)
list_memory_groups = async_to_sync(MemoryService.list_memory_groups)
