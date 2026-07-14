from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.db.models import Count, QuerySet
from django.utils import timezone

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.files.common import compute_file_hash
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
from django_ai_sdk.memories.tasks import PIPELINE_TIMEOUT_SECONDS, process_document_upload
from django_ai_sdk.permissions import (
    ConflictError,
    Operation,
    PermissionDomain,
    PermissionsMixin,
    get_assistant_permissions,
    has_perms,
)
from django_ai_sdk.tasks import TaskStatus, aget_task_status

# Catches the worker dying outright (nothing left to hit PIPELINE_TIMEOUT_SECONDS'
# own except block), so this can only ever fire after that would already have.
STALE_PROCESSING_DEADLINE_SECONDS = PIPELINE_TIMEOUT_SECONDS + 60

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


async def _fail_orphaned_processing_document(
    entry_doc: EntryDocument, task_status: TaskStatus | None
) -> None:
    """Mark a stuck PROCESSING document FAILED: either its task already
    finished without us noticing, or it's been RUNNING too long (dead worker).
    """
    if entry_doc.processing_status != EntryDocument.ProcessingStatus.PROCESSING:
        return

    task_finished = task_status.status in ("FAILED", "SUCCESSFUL") if task_status else False

    if task_finished:
        if task_status.errors:
            task_error = task_status.errors[0]
            tb_lines = task_error.traceback.strip().splitlines()
            error = tb_lines[-1] if tb_lines else task_error.type
        else:
            error = "Task finished without updating the document"
    else:
        reference_time = (task_status.started_at if task_status else None) or entry_doc.updated_at
        stale_after = reference_time + timedelta(seconds=STALE_PROCESSING_DEADLINE_SECONDS)
        if timezone.now() < stale_after:
            return
        error = "Processing did not complete and the worker never reported back"

    # Only writes if still PROCESSING, so a concurrent finish/cancel wins.
    updated = await EntryDocument.objects.filter(
        id=entry_doc.id, processing_status=EntryDocument.ProcessingStatus.PROCESSING
    ).aupdate(
        processing_status=EntryDocument.ProcessingStatus.FAILED,
        processing_error=error,
        processing_step=None,
        updated_at=timezone.now(),
    )
    if updated:
        entry_doc.processing_status = EntryDocument.ProcessingStatus.FAILED
        entry_doc.processing_error = error
        entry_doc.processing_step = None


def _document_status_out(
    entry_doc: EntryDocument, task_status: TaskStatus | None
) -> DocumentStatusOut:
    return DocumentStatusOut(
        id=str(entry_doc.id),
        status=entry_doc.processing_status,
        error=entry_doc.processing_error,
        processing_step=entry_doc.processing_step,
        task=task_status,
    )


class MemoryService(PermissionsMixin):
    """
    Service for memory operations.

    All methods are async. Use the sync-prefixed aliases for sync contexts
    (e.g., DRF class-based views).
    """

    domain = PermissionDomain.MEMORY

    # ============================================================================
    # Memory CRUD
    # ============================================================================

    @classmethod
    async def create_memory(
        cls,
        name: str,
        description: str = "",
        slug: str = "",
        is_public: bool = True,
        *,
        user: UserType,
    ) -> MemoryOut:
        """Create a new memory."""
        await cls.has_perms(user, Operation.CREATE_MEMORY)

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

    @classmethod
    async def list_memory_users(cls, memory_id: str, *, user: UserType) -> list[MemoryUserOut]:
        """List all users of a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.VIEW_MEMORY, memory)
        return [
            MemoryUserOut(
                user_id=str(o.user_id),
                can_manage=o.can_manage,
                created_at=o.created_at.isoformat(),
            )
            async for o in memory.memory_users.all().select_related("user")
        ]

    @classmethod
    async def add_memory_user(
        cls,
        memory_id: str,
        user_id: str,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> MemoryUserOut:
        """Add a user to a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
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

    @classmethod
    async def update_memory_user(
        cls,
        memory_id: str,
        user_id: str,
        can_manage: bool,
        *,
        user: UserType,
    ) -> MemoryUserOut:
        """Update a memory user's can_manage flag."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
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

    @classmethod
    async def remove_memory_user(cls, memory_id: str, user_id: str, *, user: UserType) -> None:
        """Remove a user from a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
        ownership = await _aget_or_not_found(memory.memory_users, user_id=user_id)
        await ownership.adelete()

    @classmethod
    async def add_memory_group(
        cls,
        memory_id: str,
        group_id: int,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> MemoryGroupOut:
        from django.contrib.auth.models import Group

        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
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

    @classmethod
    async def remove_memory_group(
        cls,
        memory_id: str,
        group_id: int,
        *,
        user: UserType,
    ) -> None:
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
        await memory.memory_groups.filter(group_id=group_id).adelete()

    @classmethod
    async def list_memory_groups(
        cls,
        memory_id: str,
        *,
        user: UserType,
    ) -> list[MemoryGroupOut]:
        """List all groups of a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.VIEW_MEMORY, memory)
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

    @classmethod
    async def list_memories(cls, *, user: UserType) -> list[MemoryOut]:
        """List memories visible to the requesting user."""
        qs = cls.has_queryset_perms(
            user,
            Operation.VIEW_MEMORY,
            queryset=Memory.objects.filter(is_hidden=False),
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

    @classmethod
    async def _get_assistant_memories(cls, assistant_id: str) -> QuerySet[Memory]:
        """Resolve the assistant's configured memories as a queryset."""
        try:
            assistant = await AssistantService.get(assistant_id)
        except ValueError:
            return Memory.objects.none()
        slugs: list[str] = getattr(assistant, "memories", [])
        if not slugs:
            return Memory.objects.none()
        return Memory.objects.filter(slug__in=slugs).distinct()

    @classmethod
    async def get_assistant_memories(cls, assistant_id: str) -> list[str]:
        qs = await cls._get_assistant_memories(assistant_id)
        return [str(m.id) async for m in qs]

    @classmethod
    async def get_assistant_memories_for_user(
        cls, assistant_id: str, *, user: UserType
    ) -> list[Memory]:
        """Return the assistant's configured memories the user can ``VIEW_MEMORY``.

        Filters out memories the user doesn't have permission to see, so the
        caller never receives unreadable memories.
        """
        qs = await cls._get_assistant_memories(assistant_id)
        qs = cls.has_queryset_perms(user, Operation.VIEW_MEMORY, queryset=qs)
        return [m async for m in qs]

    @classmethod
    async def link_memories(
        cls,
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
        qs = await cls._get_assistant_memories(assistant_id)
        qs = cls.has_queryset_perms(user, Operation.LINK_MEMORY, queryset=qs)
        async for memory in qs:
            await ThreadMemory.objects.aget_or_create(thread_id=thread_id, memory=memory)

    @classmethod
    async def unlink_memories(
        cls,
        assistant_id: str,
        thread_id: str,
        *,
        user: UserType,
    ) -> None:
        """Unlink an assistant's configured memories from a thread.

        Only unlinks memories the user has ``UNLINK_MEMORY`` permission for.
        """
        qs = await cls._get_assistant_memories(assistant_id)
        qs = cls.has_queryset_perms(user, Operation.UNLINK_MEMORY, queryset=qs)
        if not await qs.aexists():
            return
        await ThreadMemory.objects.filter(
            thread_id=thread_id,
            memory_id__in=[m async for m in qs.values_list("id", flat=True)],
        ).adelete()

    @classmethod
    async def get_memory(cls, memory_id: str, *, user: UserType) -> MemoryOut:
        """Get a single memory by ID."""
        memory = await _aget_or_not_found(
            Memory.objects.annotate(document_count=Count("entries")), id=memory_id
        )
        await cls.has_perms(user, Operation.VIEW_MEMORY, memory)
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

    @classmethod
    async def update_memory(
        cls,
        memory_id: str,
        name: str,
        description: str = "",
        is_public: bool | None = None,
        *,
        user: UserType,
    ) -> MemoryOut:
        """Update a memory."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPDATE_MEMORY, memory)
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

    @classmethod
    async def delete_memory(cls, memory_id: str, *, user: UserType) -> None:
        """Delete a memory and all its entries."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.DELETE_MEMORY, memory)
        await memory.adelete()

    # ============================================================================
    # Document CRUD (per memory)
    # ============================================================================

    @classmethod
    async def upload_document(
        cls, memory_id: str, file: File, *, user: UserType
    ) -> DocumentUploadResponse:
        """Save file and enqueue pipeline processing. Returns immediately with doc_id."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UPLOAD_DOCUMENT, memory)

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)
        file_hash = compute_file_hash(file)

        dup = (
            await EntryDocument.objects.filter(
                memory=memory,
                file_hash=file_hash,
            )
            .exclude(
                processing_status=EntryDocument.ProcessingStatus.FAILED,
            )
            .afirst()
        )
        if dup is not None:
            raise ConflictError("File already exists in this memory")

        entry_doc = await EntryDocument.objects.acreate(
            entry=None,
            memory=memory,
            file=file,
            file_name=file_name,
            file_hash=file_hash,
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

        return DocumentUploadResponse(
            id=str(entry_doc.id), status="processing", task_id=str(task_result.id)
        )

    @classmethod
    async def list_documents(cls, memory_id: str, *, user: UserType) -> list[DocumentOut]:
        """List all file-backed documents in a memory (all processing statuses)."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.LIST_DOCUMENTS, memory)
        entry_docs = (
            EntryDocument.objects.filter(memory_id=memory_id)
            .select_related("entry")
            .order_by("-created_at")
        )
        return [cls._entry_doc_to_out(ed) async for ed in entry_docs]

    @classmethod
    async def get_document(cls, memory_id: str, doc_id: str, *, user: UserType) -> DocumentOut:
        """Get a single document from a memory by its EntryDocument id."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.VIEW_DOCUMENT, memory)
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("entry"),
            id=doc_id,
            memory_id=memory_id,
        )
        return cls._entry_doc_to_out(entry_doc)

    @classmethod
    async def delete_document(cls, memory_id: str, doc_id: str, *, user: UserType) -> None:
        """Delete a document from a memory by its EntryDocument id.

        Works for in-flight docs (no Entry yet). When an Entry exists, deleting it
        cascades to the EntryDocument and triggers RAG cleanup via the delete signal.
        """
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.DELETE_DOCUMENT, memory)
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

    @classmethod
    async def link_memory_to_thread(cls, memory_id: str, thread_id: str, *, user: UserType) -> None:
        """Link a memory to a thread."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.LINK_MEMORY, memory)
        thread = await _aget_or_not_found(Thread.objects, id=thread_id)
        await ThreadMemory.objects.aget_or_create(
            thread=thread,
            memory=memory,
        )

    @classmethod
    async def unlink_memory_from_thread(
        cls, memory_id: str, thread_id: str, *, user: UserType
    ) -> None:
        """Unlink a memory from a thread."""
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UNLINK_MEMORY, memory)
        link = await _aget_or_not_found(
            ThreadMemory.objects, memory_id=memory_id, thread_id=thread_id
        )
        await link.adelete()

    @classmethod
    async def list_thread_memories(cls, thread_id: str, *, user: UserType) -> list[ThreadMemoryOut]:
        """List all memories connected to a thread with their active status."""
        thread_memories = (
            ThreadMemory.objects.filter(thread_id=thread_id, memory__is_hidden=False)
            .select_related("memory")
            .annotate(document_count=Count("memory__entries"))
        )

        memories = []
        async for tm in thread_memories:
            if not await cls.has_perms(
                user,
                Operation.LIST_THREAD_MEMORIES,
                tm.memory,
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

    @classmethod
    async def get_thread_memories(cls, thread_id: str, *, user: UserType) -> list[Memory]:
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
            if not await cls.has_perms(
                user,
                Operation.VIEW_MEMORY,
                tm.memory,
                raise_on_deny=False,
            ):
                continue
            result.append(tm.memory)
        return result

    @classmethod
    async def bulk_connect_memories(
        cls, thread_id: str, memory_ids: list[str], *, user: UserType
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
            await cls.has_perms(user, Operation.LINK_MEMORY, memory)
            links.append(ThreadMemory(thread=thread, memory=memory, active=True))

        await ThreadMemory.objects.abulk_create(links, ignore_conflicts=True)

        return await cls.list_thread_memories(thread_id, user=user)

    @classmethod
    async def toggle_memory_active(
        cls,
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
        await cls.has_perms(user, Operation.LINK_MEMORY, memory)
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

    @classmethod
    async def disconnect_memory_from_thread(
        cls, thread_id: str, memory_id: str, *, user: UserType
    ) -> None:
        """Disconnect a memory from a thread."""
        link = await _aget_or_not_found(
            ThreadMemory.objects, thread_id=thread_id, memory_id=memory_id
        )
        memory = await _aget_or_not_found(Memory.objects, id=memory_id)
        await cls.has_perms(user, Operation.UNLINK_MEMORY, memory)
        await link.adelete()

    # ============================================================================
    # Thread file uploads
    # ============================================================================

    @classmethod
    async def get_or_create_thread_file_memory(cls, thread_id: str) -> Memory:
        """Get or auto-create the hidden file-upload Memory for a thread."""
        thread = await _aget_or_not_found(
            Thread.objects.select_related("file_memory"), id=thread_id
        )
        if not thread.file_memory:
            memory, created = await Memory.objects.aget_or_create(
                name=f"thread_files_{thread_id}",
                defaults={"description": "Thread file uploads", "is_hidden": True},
            )
            thread.file_memory = memory
            await thread.asave(update_fields=["file_memory", "updated_at"])
            if created:
                await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)
        return thread.file_memory

    @classmethod
    async def upload_thread_file(
        cls, thread_id: str, file: File, *, user: UserType
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
        memory = await cls.get_or_create_thread_file_memory(thread_id)

        file_name = file.name or ""
        _, ext = os.path.splitext(file_name)
        file_hash = compute_file_hash(file)

        dup = (
            await EntryDocument.objects.filter(
                memory=memory,
                file_hash=file_hash,
            )
            .exclude(
                processing_status=EntryDocument.ProcessingStatus.FAILED,
            )
            .afirst()
        )
        if dup is not None:
            raise ConflictError("File already exists in this memory")

        entry_doc = await EntryDocument.objects.acreate(
            entry=None,
            memory=memory,
            file=file,
            file_name=file_name,
            file_hash=file_hash,
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

        return DocumentUploadResponse(
            id=str(entry_doc.id), status="processing", task_id=str(task_result.id)
        )

    @classmethod
    async def list_thread_files(cls, thread_id: str, *, user: UserType) -> list[DocumentOut]:
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
        return [cls._entry_doc_to_out(ed) async for ed in entry_docs]

    @classmethod
    async def delete_thread_file(cls, thread_id: str, doc_id: str, *, user: UserType) -> None:
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

    @classmethod
    async def get_document_status(cls, doc_id: str, *, user: UserType) -> DocumentStatusOut:
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
            await cls.has_perms(
                user,
                Operation.VIEW_DOCUMENT,
                entry_doc.memory,
            )
        task_status = None
        if entry_doc.task_id:
            try:
                task_status = await aget_task_status(entry_doc.task_id)
            except Exception:
                pass
        await _fail_orphaned_processing_document(entry_doc, task_status)
        return _document_status_out(entry_doc, task_status)

    @classmethod
    async def get_task_status(cls, task_id: str, *, user: UserType) -> DocumentStatusOut:
        """Return processing status for a document, looked up by its task id.

        Same permission model and response shape as ``get_document_status`` — this
        is just an alternate lookup key for consumers that only have a ``task_id``
        (e.g. straight off ``DocumentUploadResponse``), not a ``doc_id``.
        """
        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("memory"), task_id=task_id
        )
        if entry_doc.memory_id is not None:
            await cls.has_perms(
                user,
                Operation.VIEW_DOCUMENT,
                entry_doc.memory,
            )
        try:
            task_status = await aget_task_status(task_id)
        except Exception:
            task_status = None
        await _fail_orphaned_processing_document(entry_doc, task_status)
        return _document_status_out(entry_doc, task_status)

    @classmethod
    async def retry_document(cls, doc_id: str, *, user: UserType) -> DocumentStatusOut:
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
            EntryDocument.ProcessingStatus.CANCELLED,
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
            await cls.has_perms(user, Operation.UPLOAD_DOCUMENT, memory)

        entry_doc.processing_status = EntryDocument.ProcessingStatus.PENDING
        entry_doc.processing_error = ""
        entry_doc.processing_step = None
        entry_doc.cancelled_at = None
        await entry_doc.asave(
            update_fields=[
                "processing_status",
                "processing_error",
                "processing_step",
                "cancelled_at",
                "updated_at",
            ]
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
        # Re-read to return the actual persisted state — the conditional aupdate above
        # may have been skipped if the worker already advanced past PENDING.
        await entry_doc.arefresh_from_db(
            fields=["processing_status", "processing_error", "processing_step", "task_id"]
        )
        return _document_status_out(entry_doc, None)

    @classmethod
    async def cancel_document(cls, doc_id: str, *, user: UserType) -> DocumentStatusOut:
        """Cancel a pending or in-progress document.

        Sets ``cancelled_at`` so a still-running pipeline stops at its next
        step boundary (cooperative — it won't interrupt a step already in
        flight) and marks the document ``CANCELLED`` immediately so the
        caller sees it right away.
        """
        _CANCELLABLE = {
            EntryDocument.ProcessingStatus.PENDING,
            EntryDocument.ProcessingStatus.PROCESSING,
        }

        entry_doc = await _aget_or_not_found(
            EntryDocument.objects.select_related("memory"), id=doc_id
        )
        memory = entry_doc.memory
        if memory is None:
            raise ValueError("Document has no associated memory")

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
            await cls.has_perms(user, Operation.UPLOAD_DOCUMENT, memory)

        now = timezone.now()
        updated = await EntryDocument.objects.filter(
            id=entry_doc.id, processing_status__in=_CANCELLABLE
        ).aupdate(
            cancelled_at=now,
            processing_status=EntryDocument.ProcessingStatus.CANCELLED,
            processing_error="Cancelled by user",
            processing_step=None,
            updated_at=now,
        )
        if not updated:
            await entry_doc.arefresh_from_db(fields=["processing_status"])
            raise ValueError(
                f"Document cannot be cancelled in status {entry_doc.processing_status!r}."
            )
        entry_doc.cancelled_at = now
        entry_doc.processing_status = EntryDocument.ProcessingStatus.CANCELLED
        entry_doc.processing_error = "Cancelled by user"
        entry_doc.processing_step = None
        return _document_status_out(entry_doc, None)

    @classmethod
    async def get_chunk_content(
        cls, entry_id: str, chunk_id: str | None, *, user: UserType
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

        await cls.has_perms(user, Operation.VIEW_DOCUMENT, entry.memory)

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

    @classmethod
    def _entry_doc_to_out(cls, entry_doc: EntryDocument) -> DocumentOut:
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
            processing_step=entry_doc.processing_step,
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
get_task_status = async_to_sync(MemoryService.get_task_status)
retry_document = async_to_sync(MemoryService.retry_document)
cancel_document = async_to_sync(MemoryService.cancel_document)
get_chunk_content = async_to_sync(MemoryService.get_chunk_content)
list_memory_users = async_to_sync(MemoryService.list_memory_users)
add_memory_user = async_to_sync(MemoryService.add_memory_user)
update_memory_user = async_to_sync(MemoryService.update_memory_user)
remove_memory_user = async_to_sync(MemoryService.remove_memory_user)
add_memory_group = async_to_sync(MemoryService.add_memory_group)
remove_memory_group = async_to_sync(MemoryService.remove_memory_group)
list_memory_groups = async_to_sync(MemoryService.list_memory_groups)
