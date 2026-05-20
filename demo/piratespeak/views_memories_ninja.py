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
from ninja import File, Router
from ninja.files import UploadedFile

router = Router()


@router.post("", response=MemoryOut)
async def create_memory(request: HttpRequest, payload: MemoryIn) -> MemoryOut:
    return await MemoryService.create_memory(
        name=payload.name, slug=payload.slug, description=payload.description
    )


@router.get("", response=list[MemoryOut])
async def list_memories(request: HttpRequest) -> list[MemoryOut]:
    return await MemoryService.list_memories()


@router.get("/{memory_id}", response=MemoryOut)
async def get_memory(request: HttpRequest, memory_id: str) -> MemoryOut:
    return await MemoryService.get_memory(memory_id)


@router.put("/{memory_id}", response=MemoryOut)
async def update_memory(request: HttpRequest, memory_id: str, payload: MemoryIn) -> MemoryOut:
    return await MemoryService.update_memory(
        memory_id=memory_id, name=payload.name, description=payload.description
    )


@router.delete("/{memory_id}", response={204: None})
async def delete_memory(request: HttpRequest, memory_id: str) -> tuple[int, None]:
    await MemoryService.delete_memory(memory_id)
    return 204, None


@router.post("/{memory_id}/documents", response={200: DocumentOut, 400: dict})
async def upload_document(
    request: HttpRequest,
    memory_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut | tuple[int, dict]:
    return await MemoryService.upload_document(memory_id, file)


@router.get("/{memory_id}/documents", response=list[DocumentOut])
async def list_documents(request: HttpRequest, memory_id: str) -> list[DocumentOut]:
    return await MemoryService.list_documents(memory_id)


@router.get("/{memory_id}/documents/{doc_id}", response=DocumentOut)
async def get_document(request: HttpRequest, memory_id: str, doc_id: str) -> DocumentOut:
    return await MemoryService.get_document(memory_id, doc_id)


@router.delete("/{memory_id}/documents/{doc_id}", response={204: None})
async def delete_document(request: HttpRequest, memory_id: str, doc_id: str) -> tuple[int, None]:
    await MemoryService.delete_document(memory_id, doc_id)
    return 204, None


@router.post("/{memory_id}/link/{thread_id}", response={204: None})
async def link_thread(request: HttpRequest, memory_id: str, thread_id: str) -> tuple[int, None]:
    await MemoryService.link_memory_to_thread(memory_id, thread_id)
    return 204, None


@router.delete("/{memory_id}/link/{thread_id}", response={204: None})
async def unlink_thread(request: HttpRequest, memory_id: str, thread_id: str) -> tuple[int, None]:
    await MemoryService.unlink_memory_from_thread(memory_id, thread_id)
    return 204, None


@router.get("/thread/{thread_id}", response=list[ThreadMemoryOut])
async def list_thread_memories(request: HttpRequest, thread_id: str) -> list[ThreadMemoryOut]:
    return await MemoryService.list_thread_memories(thread_id)


@router.post("/thread/{thread_id}/bulk", response=list[ThreadMemoryOut])
async def bulk_connect_memories(
    request: HttpRequest, thread_id: str, payload: BulkConnectMemoriesIn
) -> list[ThreadMemoryOut]:
    return await MemoryService.bulk_connect_memories(thread_id, payload.memory_ids)


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


@router.patch("/thread/{thread_id}/{memory_id}", response=ThreadMemoryOut)
async def toggle_memory_active(
    request: HttpRequest, thread_id: str, memory_id: str, payload: ToggleMemoryActiveIn
) -> ThreadMemoryOut:
    return await MemoryService.toggle_memory_active(thread_id, memory_id, payload.active)


@router.delete("/thread/{thread_id}/{memory_id}", response={204: None})
async def disconnect_memory_from_thread(
    request: HttpRequest, thread_id: str, memory_id: str
) -> tuple[int, None]:
    await MemoryService.disconnect_memory_from_thread(thread_id, memory_id)
    return 204, None
