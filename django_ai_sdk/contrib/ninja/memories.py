"""Memories (knowledge bases), their documents, sharing, and thread links/files."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from django.http import HttpRequest
from ninja import File, Router
from ninja.files import UploadedFile

from django_ai_sdk.contrib.ninja.routing import ERRORS, ApiRouter, Limit, Offset
from django_ai_sdk.contrib.ninja.schemas import (
    MemoryOutResponse,
    SourceContentOut,
    UploadSettingsOut,
)
from django_ai_sdk.errors import NotFound
from django_ai_sdk.memories.schemas import (
    DocumentOut,
    DocumentStatusOut,
    DocumentUploadResponse,
    MemoryGroupOut,
    MemoryOut,
    MemoryUserOut,
    ThreadMemoryOut,
)
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.views.files import athread_file_response
from django_ai_sdk.views.permissions import amemory_permissions
from django_ai_sdk.views.schemas import (
    AddMemoryGroupIn,
    AddMemoryUserIn,
    BulkConnectMemoriesIn,
    MemoryIn,
    ToggleMemoryActiveIn,
    UpdateMemoryUserIn,
)

routes = ApiRouter()

NO_CONTENT = {204: None, **ERRORS}
ACCEPTED = {202: DocumentUploadResponse, **ERRORS, 409: ERRORS[400]}


@routes.get("/settings", response=UploadSettingsOut)
async def get_upload_settings(request: HttpRequest) -> Any:
    from django_ai_sdk import files

    result = files.get_upload_settings()
    return UploadSettingsOut(
        max_upload_size=result.max_upload_size, allowed_mime_types=result.allowed_mime_types
    )


@routes.post("", response=MemoryOut)
async def create_memory(request: HttpRequest, payload: MemoryIn) -> Any:
    return await MemoryService.create_memory(
        name=payload.name,
        slug=payload.slug,
        description=payload.description,
        is_public=payload.is_public,
        user=request.user,
    )


@routes.get("", response=list[MemoryOutResponse])
async def list_memories(
    request: HttpRequest,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    memories = await MemoryService.list_memories(user=request.user, limit=limit, offset=offset)
    return [
        MemoryOutResponse(
            **m.model_dump(), permissions=await amemory_permissions(request.user, m.id)
        )
        for m in memories
    ]


@routes.get("/{memory_id}", response=MemoryOutResponse)
async def get_memory(request: HttpRequest, memory_id: str) -> Any:
    memory = await MemoryService.get_memory(memory_id, user=request.user)
    perms = await amemory_permissions(request.user, memory_id)
    return MemoryOutResponse(**memory.model_dump(), permissions=perms)


@routes.put("/{memory_id}", response=MemoryOut)
async def update_memory(request: HttpRequest, memory_id: str, payload: MemoryIn) -> Any:
    return await MemoryService.update_memory(
        memory_id=memory_id,
        name=payload.name,
        description=payload.description,
        is_public=payload.is_public,
        user=request.user,
    )


@routes.delete("/{memory_id}", response=NO_CONTENT)
async def delete_memory(request: HttpRequest, memory_id: str) -> Any:
    await MemoryService.delete_memory(memory_id, user=request.user)
    return 204, None


@routes.post("/{memory_id}/documents", response=ACCEPTED)
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: File[UploadedFile],
) -> Any:
    return 202, await MemoryService.upload_document(memory_id, file, user=request.user)


@routes.get("/{memory_id}/documents/{doc_id}/status", response=DocumentStatusOut)
async def get_document_status(request: HttpRequest, memory_id: str, doc_id: str) -> Any:
    return await MemoryService.get_document_status(doc_id, user=request.user)


@routes.get("/{memory_id}/documents", response=list[DocumentOut])
async def list_documents(
    request: HttpRequest,
    memory_id: str,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    return await MemoryService.list_documents(
        memory_id, user=request.user, limit=limit, offset=offset
    )


@routes.get("/{memory_id}/documents/{doc_id}", response=DocumentOut)
async def get_document(request: HttpRequest, memory_id: str, doc_id: str) -> Any:
    return await MemoryService.get_document(memory_id, doc_id, user=request.user)


@routes.delete("/{memory_id}/documents/{doc_id}", response=NO_CONTENT)
async def delete_document(request: HttpRequest, memory_id: str, doc_id: str) -> Any:
    await MemoryService.delete_document(memory_id, doc_id, user=request.user)
    return 204, None


@routes.post("/{memory_id}/link/{thread_id}", response=NO_CONTENT)
async def link_thread(request: HttpRequest, memory_id: str, thread_id: str) -> Any:
    await MemoryService.link_memory_to_thread(memory_id, thread_id, user=request.user)
    return 204, None


@routes.delete("/{memory_id}/link/{thread_id}", response=NO_CONTENT)
async def unlink_thread(request: HttpRequest, memory_id: str, thread_id: str) -> Any:
    await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
    return 204, None


@routes.get("/thread/{thread_id}", response=list[ThreadMemoryOut])
async def list_thread_memories(
    request: HttpRequest,
    thread_id: str,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    return await MemoryService.list_thread_memories(
        thread_id, user=request.user, limit=limit, offset=offset
    )


@routes.post("/thread/{thread_id}/bulk", response=list[ThreadMemoryOut])
async def bulk_connect_memories(
    request: HttpRequest, thread_id: str, payload: BulkConnectMemoriesIn
) -> Any:
    return await MemoryService.bulk_connect_memories(
        thread_id, payload.memory_ids, user=request.user
    )


@routes.post("/thread/{thread_id}/files", response=ACCEPTED)
async def upload_thread_file(
    request: HttpRequest,
    thread_id: str,
    file: File[UploadedFile],
) -> Any:
    return 202, await MemoryService.upload_thread_file(thread_id, file, user=request.user)


@routes.get("/thread/{thread_id}/files/{doc_id}/status", response=DocumentStatusOut)
async def get_thread_file_status(request: HttpRequest, thread_id: str, doc_id: str) -> Any:
    return await MemoryService.get_document_status(doc_id, user=request.user)


@routes.get("/thread/{thread_id}/files/{doc_id}/download")
async def download_thread_file(request: HttpRequest, thread_id: str, doc_id: str) -> Any:
    """The file's bytes. Only raster images render inline; everything else downloads."""
    return await athread_file_response(thread_id, doc_id, user=request.user)


@routes.get("/thread/{thread_id}/files", response=list[DocumentOut])
async def list_thread_files(
    request: HttpRequest,
    thread_id: str,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    return await MemoryService.list_thread_files(
        thread_id, user=request.user, limit=limit, offset=offset
    )


@routes.delete("/thread/{thread_id}/files/{doc_id}", response=NO_CONTENT)
async def delete_thread_file(request: HttpRequest, thread_id: str, doc_id: str) -> Any:
    await MemoryService.delete_thread_file(thread_id, doc_id, user=request.user)
    return 204, None


@routes.patch("/thread/{thread_id}/{memory_id}", response=ThreadMemoryOut)
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> Any:
    return await MemoryService.toggle_memory_active(
        thread_id, memory_id, payload.active, user=request.user
    )


@routes.delete("/thread/{thread_id}/{memory_id}", response=NO_CONTENT)
async def disconnect_memory_from_thread(
    request: HttpRequest, thread_id: str, memory_id: str
) -> Any:
    await MemoryService.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
    return 204, None


@routes.get("/{memory_id}/users/", response=list[MemoryUserOut])
async def list_memory_users(
    request: HttpRequest,
    memory_id: str,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    return await MemoryService.list_memory_users(
        memory_id, user=request.user, limit=limit, offset=offset
    )


@routes.post("/{memory_id}/users/", response=MemoryUserOut)
async def add_memory_user(request: HttpRequest, memory_id: str, payload: AddMemoryUserIn) -> Any:
    return await MemoryService.add_memory_user(
        memory_id, payload.user_id, payload.can_manage, user=request.user
    )


@routes.patch("/{memory_id}/users/{user_id}/", response=MemoryUserOut)
async def update_memory_user(
    request: HttpRequest, memory_id: str, user_id: str, payload: UpdateMemoryUserIn
) -> Any:
    return await MemoryService.update_memory_user(
        memory_id, user_id, payload.can_manage, user=request.user
    )


@routes.delete("/{memory_id}/users/{user_id}/", response=NO_CONTENT)
async def remove_memory_user(request: HttpRequest, memory_id: str, user_id: str) -> Any:
    await MemoryService.remove_memory_user(memory_id, user_id, user=request.user)
    return 204, None


@routes.get("/{memory_id}/groups/", response=list[MemoryGroupOut])
async def list_memory_groups(request: HttpRequest, memory_id: str) -> Any:
    return await MemoryService.list_memory_groups(memory_id, user=request.user)


@routes.post("/{memory_id}/groups/", response=MemoryGroupOut)
async def add_memory_group(request: HttpRequest, memory_id: str, payload: AddMemoryGroupIn) -> Any:
    return await MemoryService.add_memory_group(
        memory_id, payload.group_id, payload.can_manage, user=request.user
    )


@routes.delete("/{memory_id}/groups/{group_id}/", response=NO_CONTENT)
async def remove_memory_group(request: HttpRequest, memory_id: str, group_id: int) -> Any:
    await MemoryService.remove_memory_group(memory_id, group_id, user=request.user)
    return 204, None


@routes.get("/source/{entry_id}/{chunk_id}", response=SourceContentOut)
async def get_source_content(request: HttpRequest, entry_id: str, chunk_id: str) -> Any:
    content = await MemoryService.get_chunk_content(entry_id, chunk_id or None, user=request.user)
    if content is None:
        raise NotFound(f"Entry not found: {entry_id}")
    return SourceContentOut(content=content)


def get_memories_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the memory endpoints; mount with ``api.add_router("/memories", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
