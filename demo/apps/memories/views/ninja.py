from __future__ import annotations

from django.http import HttpRequest
from django_ai_sdk.memories.schemas import (
    AddMemoryGroupIn,
    AddMemoryUserIn,
    BulkConnectMemoriesIn,
    DocumentOut,
    DocumentStatusOut,
    DocumentUploadResponse,
    MemoryGroupOut,
    MemoryIn,
    MemoryOut,
    MemoryUserOut,
    ThreadMemoryOut,
    ToggleMemoryActiveIn,
    UpdateMemoryUserIn,
)
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import ConflictError, ObjectPermissions, PermissionDenied
from ninja import File, Router, Schema
from ninja.files import UploadedFile

from .permissions import memory_permissions

router = Router()


class MemoryOutResponse(MemoryOut):
    permissions: ObjectPermissions = ObjectPermissions()


class SourceContentOut(Schema):
    content: str


class Error(Schema):
    message: str
    code: int | None = None


class UploadSettingsOut(Schema):
    max_upload_size: int
    allowed_mime_types: list[str]


@router.get(
    "/settings",
    response=UploadSettingsOut,
    operation_id="get_upload_settings",
)
async def get_upload_settings(request: HttpRequest) -> UploadSettingsOut:
    from django_ai_sdk.files import get_upload_settings as _get_upload_settings

    result = _get_upload_settings()
    return UploadSettingsOut(
        max_upload_size=result.max_upload_size,
        allowed_mime_types=result.allowed_mime_types,
    )


@router.post("", response={200: MemoryOut, 403: dict}, operation_id="create_memory")
async def create_memory(request: HttpRequest, payload: MemoryIn) -> MemoryOut | tuple[int, dict]:
    try:
        return await MemoryService.create_memory(
            name=payload.name,
            slug=payload.slug,
            description=payload.description,
            is_public=payload.is_public,
            user=request.user,
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("", response={200: list[MemoryOutResponse], 403: dict}, operation_id="list_memories")
async def list_memories(
    request: HttpRequest, limit: int = 100, offset: int = 0
) -> list[MemoryOutResponse] | tuple[int, dict]:
    try:
        memories = await MemoryService.list_memories(user=request.user, limit=limit, offset=offset)
        result = []
        for m in memories:
            perms = await memory_permissions(request.user, m.id)
            result.append(MemoryOutResponse(**m.model_dump(), permissions=perms))
        return result
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("/{memory_id}", response={200: MemoryOutResponse, 403: dict}, operation_id="get_memory")
async def get_memory(request: HttpRequest, memory_id: str) -> MemoryOutResponse | tuple[int, dict]:
    try:
        memory = await MemoryService.get_memory(memory_id, user=request.user)
        perms = await memory_permissions(request.user, memory_id)
        return MemoryOutResponse(**memory.model_dump(), permissions=perms)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.put("/{memory_id}", response={200: MemoryOut, 403: dict}, operation_id="update_memory")
async def update_memory(
    request: HttpRequest, memory_id: str, payload: MemoryIn
) -> MemoryOut | tuple[int, dict]:
    try:
        return await MemoryService.update_memory(
            memory_id=memory_id,
            name=payload.name,
            description=payload.description,
            is_public=payload.is_public,
            user=request.user,
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete("/{memory_id}", response={204: None, 403: dict}, operation_id="delete_memory")
async def delete_memory(request: HttpRequest, memory_id: str) -> tuple[int, None | dict]:
    try:
        await MemoryService.delete_memory(memory_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/{memory_id}/documents",
    response={202: DocumentUploadResponse, 400: dict, 403: dict, 409: dict},
    operation_id="upload_document",
)
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> tuple[int, DocumentUploadResponse | dict]:
    try:
        return 202, await MemoryService.upload_document(memory_id, file, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}
    except ConflictError as e:
        return 409, {"detail": str(e)}


@router.get(
    "/{memory_id}/documents/{doc_id}/status",
    response={200: DocumentStatusOut, 403: dict},
    operation_id="get_document_status",
)
async def get_document_status(
    request: HttpRequest, memory_id: str, doc_id: str
) -> DocumentStatusOut | tuple[int, dict]:
    try:
        return await MemoryService.get_document_status(doc_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/{memory_id}/documents",
    response={200: list[DocumentOut], 403: dict},
    operation_id="list_documents",
)
async def list_documents(
    request: HttpRequest, memory_id: str, limit: int = 100, offset: int = 0
) -> list[DocumentOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_documents(
            memory_id, user=request.user, limit=limit, offset=offset
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/{memory_id}/documents/{doc_id}",
    response={200: DocumentOut, 403: dict},
    operation_id="get_document",
)
async def get_document(
    request: HttpRequest, memory_id: str, doc_id: str
) -> DocumentOut | tuple[int, dict]:
    try:
        return await MemoryService.get_document(memory_id, doc_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete(
    "/{memory_id}/documents/{doc_id}",
    response={204: None, 403: dict},
    operation_id="delete_document",
)
async def delete_document(
    request: HttpRequest, memory_id: str, doc_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.delete_document(memory_id, doc_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/{memory_id}/link/{thread_id}", response={204: None, 403: dict}, operation_id="link_thread"
)
async def link_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.link_memory_to_thread(memory_id, thread_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete(
    "/{memory_id}/link/{thread_id}", response={204: None, 403: dict}, operation_id="unlink_thread"
)
async def unlink_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/thread/{thread_id}",
    response={200: list[ThreadMemoryOut], 403: dict},
    operation_id="list_thread_memories",
)
async def list_thread_memories(
    request: HttpRequest, thread_id: str, limit: int = 100, offset: int = 0
) -> list[ThreadMemoryOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_thread_memories(
            thread_id, user=request.user, limit=limit, offset=offset
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/thread/{thread_id}/bulk",
    response={200: list[ThreadMemoryOut], 403: dict},
    operation_id="bulk_connect_memories",
)
async def bulk_connect_memories(
    request: HttpRequest, thread_id: str, payload: BulkConnectMemoriesIn
) -> list[ThreadMemoryOut] | tuple[int, dict]:
    try:
        return await MemoryService.bulk_connect_memories(
            thread_id, payload.memory_ids, user=request.user
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/thread/{thread_id}/files",
    response={202: DocumentUploadResponse, 400: dict, 409: dict},
    operation_id="upload_thread_file",
)
async def upload_thread_file(
    request: HttpRequest,
    thread_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> tuple[int, DocumentUploadResponse | dict]:
    try:
        return 202, await MemoryService.upload_thread_file(thread_id, file, user=request.user)
    except ConflictError as e:
        return 409, {"detail": str(e)}


@router.get(
    "/thread/{thread_id}/files/{doc_id}/status",
    response={200: DocumentStatusOut},
    operation_id="get_thread_file_status",
)
async def get_thread_file_status(
    request: HttpRequest, thread_id: str, doc_id: str
) -> DocumentStatusOut:
    return await MemoryService.get_document_status(doc_id, user=request.user)


@router.get(
    "/thread/{thread_id}/files", response=list[DocumentOut], operation_id="list_thread_files"
)
async def list_thread_files(
    request: HttpRequest, thread_id: str, limit: int = 100, offset: int = 0
) -> list[DocumentOut]:
    return await MemoryService.list_thread_files(
        thread_id, user=request.user, limit=limit, offset=offset
    )


@router.delete(
    "/thread/{thread_id}/files/{doc_id}", response={204: None}, operation_id="delete_thread_file"
)
async def delete_thread_file(request: HttpRequest, thread_id: str, doc_id: str) -> tuple[int, None]:
    await MemoryService.delete_thread_file(thread_id, doc_id, user=request.user)
    return 204, None


@router.patch(
    "/thread/{thread_id}/{memory_id}",
    response={200: ThreadMemoryOut, 403: dict},
    operation_id="toggle_memory_active",
)
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> ThreadMemoryOut | tuple[int, dict]:
    try:
        return await MemoryService.toggle_memory_active(
            thread_id, memory_id, payload.active, user=request.user
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete(
    "/thread/{thread_id}/{memory_id}",
    response={204: None, 403: dict},
    operation_id="disconnect_memory_from_thread",
)
async def disconnect_memory_from_thread(
    request: HttpRequest, thread_id: str, memory_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/{memory_id}/users/",
    response={200: list[MemoryUserOut], 403: dict, 404: dict},
    operation_id="list_memory_users",
)
async def list_memory_users(
    request: HttpRequest, memory_id: str, limit: int = 100, offset: int = 0
) -> list[MemoryUserOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_memory_users(
            memory_id, user=request.user, limit=limit, offset=offset
        )
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/{memory_id}/users/",
    response={200: MemoryUserOut, 403: dict, 404: dict},
    operation_id="add_memory_user",
)
async def add_memory_user(
    request: HttpRequest, memory_id: str, payload: AddMemoryUserIn
) -> MemoryUserOut | tuple[int, dict]:
    try:
        return await MemoryService.add_memory_user(
            memory_id, payload.user_id, payload.can_manage, user=request.user
        )
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.patch(
    "/{memory_id}/users/{user_id}/",
    response={200: MemoryUserOut, 403: dict, 404: dict},
    operation_id="update_memory_user",
)
async def update_memory_user(
    request: HttpRequest, memory_id: str, user_id: str, payload: UpdateMemoryUserIn
) -> MemoryUserOut | tuple[int, dict]:
    try:
        return await MemoryService.update_memory_user(
            memory_id, user_id, payload.can_manage, user=request.user
        )
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete(
    "/{memory_id}/users/{user_id}/",
    response={204: None, 403: dict, 404: dict},
    operation_id="remove_memory_user",
)
async def remove_memory_user(
    request: HttpRequest, memory_id: str, user_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.remove_memory_user(memory_id, user_id, user=request.user)
        return 204, None
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/{memory_id}/groups/",
    response={200: list[MemoryGroupOut], 403: dict, 404: dict},
    operation_id="list_memory_groups",
)
async def list_memory_groups(
    request: HttpRequest, memory_id: str
) -> list[MemoryGroupOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_memory_groups(memory_id, user=request.user)
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post(
    "/{memory_id}/groups/",
    response={200: MemoryGroupOut, 403: dict, 404: dict},
    operation_id="add_memory_group",
)
async def add_memory_group(
    request: HttpRequest, memory_id: str, payload: AddMemoryGroupIn
) -> MemoryGroupOut | tuple[int, dict]:
    try:
        return await MemoryService.add_memory_group(
            memory_id, payload.group_id, payload.can_manage, user=request.user
        )
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete(
    "/{memory_id}/groups/{group_id}/",
    response={204: None, 403: dict, 404: dict},
    operation_id="remove_memory_group",
)
async def remove_memory_group(
    request: HttpRequest, memory_id: str, group_id: int
) -> tuple[int, None | dict]:
    try:
        await MemoryService.remove_memory_group(memory_id, group_id, user=request.user)
        return 204, None
    except ValueError as e:
        return 404, {"detail": str(e)}
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get(
    "/source/{entry_id}/{chunk_id}",
    response={200: SourceContentOut, 404: Error},
    operation_id="get_source_content",
)
async def get_source_content(
    request: HttpRequest, entry_id: str, chunk_id: str
) -> tuple[int, Error] | SourceContentOut:
    content = await MemoryService.get_chunk_content(entry_id, chunk_id or None, user=request.user)
    if content is None:
        return 404, Error(message=f"Entry not found: {entry_id}")
    return SourceContentOut(content=content)
