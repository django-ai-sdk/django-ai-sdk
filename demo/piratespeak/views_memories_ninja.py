from django.http import HttpRequest
from django_ai_sdk.memories.schemas import (
    BulkConnectMemoriesIn,
    DocumentOut,
    MemoryIn,
    MemoryOut,
    ThreadMemoryOut,
    ToggleMemoryActiveIn,
)
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import PermissionDenied
from ninja import File, Router
from ninja.files import UploadedFile

router = Router()


@router.post("", response={200: MemoryOut, 403: dict})
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


@router.get("", response={200: list[MemoryOut], 403: dict})
async def list_memories(request: HttpRequest) -> list[MemoryOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_memories(user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("/{memory_id}", response={200: MemoryOut, 403: dict})
async def get_memory(request: HttpRequest, memory_id: str) -> MemoryOut | tuple[int, dict]:
    try:
        return await MemoryService.get_memory(memory_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.put("/{memory_id}", response={200: MemoryOut, 403: dict})
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


@router.delete("/{memory_id}", response={204: None, 403: dict})
async def delete_memory(request: HttpRequest, memory_id: str) -> tuple[int, None | dict]:
    try:
        await MemoryService.delete_memory(memory_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post("/{memory_id}/documents", response={200: DocumentOut, 400: dict, 403: dict})
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut | tuple[int, dict]:
    try:
        return await MemoryService.upload_document(memory_id, file, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("/{memory_id}/documents", response={200: list[DocumentOut], 403: dict})
async def list_documents(request: HttpRequest, memory_id: str) -> list[DocumentOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_documents(memory_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("/{memory_id}/documents/{doc_id}", response={200: DocumentOut, 403: dict})
async def get_document(
    request: HttpRequest, memory_id: str, doc_id: str
) -> DocumentOut | tuple[int, dict]:
    try:
        return await MemoryService.get_document(memory_id, doc_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete("/{memory_id}/documents/{doc_id}", response={204: None, 403: dict})
async def delete_document(
    request: HttpRequest, memory_id: str, doc_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.delete_document(memory_id, doc_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post("/{memory_id}/link/{thread_id}", response={204: None, 403: dict})
async def link_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.link_memory_to_thread(memory_id, thread_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete("/{memory_id}/link/{thread_id}", response={204: None, 403: dict})
async def unlink_thread(
    request: HttpRequest, memory_id: str, thread_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.get("/thread/{thread_id}", response={200: list[ThreadMemoryOut], 403: dict})
async def list_thread_memories(
    request: HttpRequest, thread_id: str
) -> list[ThreadMemoryOut] | tuple[int, dict]:
    try:
        return await MemoryService.list_thread_memories(thread_id, user=request.user)
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post("/thread/{thread_id}/bulk", response={200: list[ThreadMemoryOut], 403: dict})
async def bulk_connect_memories(
    request: HttpRequest, thread_id: str, payload: BulkConnectMemoriesIn
) -> list[ThreadMemoryOut] | tuple[int, dict]:
    try:
        return await MemoryService.bulk_connect_memories(
            thread_id, payload.memory_ids, user=request.user
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.post("/thread/{thread_id}/files", response={200: DocumentOut, 400: dict})
async def upload_thread_file(
    request: HttpRequest,
    thread_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut | tuple[int, dict]:
    return await MemoryService.upload_thread_file(thread_id, file)


@router.get("/thread/{thread_id}/files", response=list[DocumentOut])
async def list_thread_files(request: HttpRequest, thread_id: str) -> list[DocumentOut]:
    return await MemoryService.list_thread_files(thread_id)


@router.delete("/thread/{thread_id}/files/{doc_id}", response={204: None})
async def delete_thread_file(request: HttpRequest, thread_id: str, doc_id: str) -> tuple[int, None]:
    await MemoryService.delete_thread_file(thread_id, doc_id)
    return 204, None


@router.patch("/thread/{thread_id}/{memory_id}", response={200: ThreadMemoryOut, 403: dict})
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> ThreadMemoryOut | tuple[int, dict]:
    try:
        return await MemoryService.toggle_memory_active(
            thread_id, memory_id, payload.active, user=request.user
        )
    except PermissionDenied as e:
        return 403, {"detail": str(e)}


@router.delete("/thread/{thread_id}/{memory_id}", response={204: None, 403: dict})
async def disconnect_memory_from_thread(
    request: HttpRequest, thread_id: str, memory_id: str
) -> tuple[int, None | dict]:
    try:
        await MemoryService.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
        return 204, None
    except PermissionDenied as e:
        return 403, {"detail": str(e)}
