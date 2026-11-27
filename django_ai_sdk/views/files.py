from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync, sync_to_async
from django.http import FileResponse, HttpResponse

from django_ai_sdk.memories.services import MemoryService

if TYPE_CHECKING:
    from django_ai_sdk.memories.models import EntryDocument
    from django_ai_sdk.types import UserType

INLINE_FILE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


async def athread_file_response(thread_id: str, doc_id: str, *, user: UserType) -> FileResponse:
    """
    Raises:
        NotFound: No such thread or file.
        PermissionDenied: `user` may not view the thread's files.
    """
    doc = await MemoryService.get_thread_file(thread_id, doc_id, user=user)
    return await sync_to_async(_file_response)(doc)


async def asource_file_response(doc_id: str, *, user: UserType) -> FileResponse:
    """A PDF opens inline in the browser's own PDF viewer, so a link with ``#page=N``
    opens it on that page. No ``sandbox`` CSP: it stops that viewer.

    Raises:
        NotFound: No such document, or none with a preview.
        PermissionDenied: `user` may not view the memory's documents.
    """
    doc = await MemoryService.get_source_file(doc_id, user=user)
    if doc.file_extension.lower() == "pdf":
        return await sync_to_async(_pdf_response)(doc)
    return await sync_to_async(_file_response)(doc)


async def asource_page_response(
    doc_id: str, page: int, quotes: list[str], *, user: UserType
) -> HttpResponse:
    """``X-Mark`` says where the marks are, "top,left,bottom,right" as
    fractions of the page ("" when nothing is marked).

    Raises:
        NotFound: No such document or page.
        PermissionDenied: `user` may not view the memory's documents.
    """
    image, mark = await MemoryService.render_source_page(doc_id, page, quotes, user=user)
    response = HttpResponse(image, content_type="image/png")
    response["X-Mark"] = mark
    response["Cache-Control"] = "private, max-age=86400"
    response["X-Content-Type-Options"] = "nosniff"
    return response


thread_file_response = async_to_sync(athread_file_response)
source_file_response = async_to_sync(asource_file_response)
source_page_response = async_to_sync(asource_page_response)


def _file_response(doc: EntryDocument) -> FileResponse:
    inline = doc.content_type in INLINE_FILE_TYPES
    response = FileResponse(
        doc.file.open("rb"),
        as_attachment=not inline,
        filename=doc.file_name,
        content_type=doc.content_type if inline else "application/octet-stream",
    )
    response["X-Content-Type-Options"] = "nosniff"
    response["Content-Security-Policy"] = "sandbox"
    return response


def _pdf_response(doc: EntryDocument) -> FileResponse:
    response = FileResponse(
        doc.file.open("rb"),
        as_attachment=False,
        filename=doc.file_name,
        content_type="application/pdf",
    )
    response["X-Content-Type-Options"] = "nosniff"
    return response
