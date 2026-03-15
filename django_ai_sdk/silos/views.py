"""
TODO: we should use this as a reference only, for now there is no authentication.
"""

from django.db.models import Count
from django.http import HttpRequest
from ninja import File, Router
from ninja.files import UploadedFile

from django_ai_sdk.silos.models import Document, Silo, ThreadSilo
from django_ai_sdk.silos.schemas import (
    DocumentOut,
    SiloIn,
    SiloOut,
)

router = Router()


@router.post("", response=SiloOut)
async def create_silo(request: HttpRequest, payload: SiloIn) -> SiloOut:
    """Create a new silo."""
    silo = await Silo.objects.acreate(
        name=payload.name,
        description=payload.description or "",
    )
    return SiloOut(
        id=str(silo.id),
        name=silo.name,
        description=silo.description,
        document_count=0,
        created_at=silo.created_at.isoformat(),
        updated_at=silo.updated_at.isoformat(),
    )


@router.get("", response=list[SiloOut])
def list_silos(request: HttpRequest) -> list[SiloOut]:
    """List all silos."""
    silos = Silo.objects.annotate(document_count=Count("documents")).order_by("-created_at")
    return [
        SiloOut(
            id=str(silo.id),
            name=silo.name,
            description=silo.description,
            document_count=silo.document_count,
            created_at=silo.created_at.isoformat(),
            updated_at=silo.updated_at.isoformat(),
        )
        for silo in silos
    ]


@router.get("/{silo_id}", response=SiloOut)
async def get_silo(request: HttpRequest, silo_id: str) -> SiloOut:
    """Get a single silo by ID."""
    silo = await Silo.objects.annotate(document_count=Count("documents")).aget(id=silo_id)
    return SiloOut(
        id=str(silo.id),
        name=silo.name,
        description=silo.description,
        document_count=silo.document_count,
        created_at=silo.created_at.isoformat(),
        updated_at=silo.updated_at.isoformat(),
    )


@router.put("/{silo_id}", response=SiloOut)
def update_silo(request: HttpRequest, silo_id: str, payload: SiloIn) -> SiloOut:
    """Update a silo."""
    silo = Silo.objects.get(id=silo_id)
    silo.name = payload.name
    silo.description = payload.description or ""
    silo.save()
    doc_count = silo.documents.count()
    return SiloOut(
        id=str(silo.id),
        name=silo.name,
        description=silo.description,
        document_count=doc_count,
        created_at=silo.created_at.isoformat(),
        updated_at=silo.updated_at.isoformat(),
    )


@router.delete("/{silo_id}", response={204: None})
async def delete_silo(request: HttpRequest, silo_id: str) -> tuple[int, None]:
    """Delete a silo and all its documents."""
    silo = await Silo.objects.aget(id=silo_id)
    await silo.adelete()
    return 204, None


@router.post("/{silo_id}/documents", response=DocumentOut)
async def upload_document(
    request: HttpRequest,
    silo_id: str,
    file: UploadedFile = File(...),  # type: ignore
) -> DocumentOut:
    """Upload a file to a silo."""
    import os

    from django_ai_sdk.silos.utils import extract_document

    silo = await Silo.objects.aget(id=silo_id)
    file_name = file.name or ""
    _, ext = os.path.splitext(file_name)

    content = ""
    extraction = None
    if ext.lower() in (".txt", ".md"):
        content = file.read().decode("utf-8", errors="replace")
        extraction = await extract_document(content)

    doc = await Document.objects.acreate(
        silo=silo,
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


@router.get("/{silo_id}/documents", response=list[DocumentOut])
def list_documents(request: HttpRequest, silo_id: str) -> list[DocumentOut]:
    """List all documents in a silo."""
    silo = Silo.objects.get(id=silo_id)
    docs = silo.documents.all().order_by("-created_at")
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


@router.get("/{silo_id}/documents/{doc_id}", response=DocumentOut)
def get_document(request: HttpRequest, silo_id: str, doc_id: str) -> DocumentOut:
    """Get a single document from a silo."""
    doc = Document.objects.get(id=doc_id, silo_id=silo_id)
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


@router.delete("/{silo_id}/documents/{doc_id}", response={204: None})
async def delete_document(request: HttpRequest, silo_id: str, doc_id: str) -> tuple[int, None]:
    """Delete a document from a silo."""
    doc = await Document.objects.aget(id=doc_id, silo_id=silo_id)
    await doc.adelete()
    return 204, None


@router.post("/{silo_id}/link/{thread_id}", response={204: None})
async def link_thread(request: HttpRequest, silo_id: str, thread_id: str) -> tuple[int, None]:
    """Link a silo to a thread."""
    silo = await Silo.objects.aget(id=silo_id)
    from django_ai_sdk.conversation.models import Thread

    thread = await Thread.objects.aget(id=thread_id)
    await ThreadSilo.objects.aget_or_create(
        thread=thread,
        silo=silo,
    )
    return 204, None


@router.delete("/{silo_id}/link/{thread_id}", response={204: None})
async def unlink_thread(request: HttpRequest, silo_id: str, thread_id: str) -> tuple[int, None]:
    """Unlink a silo from a thread."""
    link = await ThreadSilo.objects.aget(silo_id=silo_id, thread_id=thread_id)
    await link.adelete()
    return 204, None
