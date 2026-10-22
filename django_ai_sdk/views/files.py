"""Serving a thread file's bytes, and a cited document's pages, shared by the contrib
Ninja and DRF layers.

The helpers take ids and a user, not a document: the lookup checks ``VIEW_FILE`` (see
``MemoryService.get_thread_file``) or ``VIEW_DOCUMENT`` (``MemoryService.get_source_file``),
so there is no way to serve a file unchecked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync, sync_to_async
from django.http import FileResponse, HttpResponse

from django_ai_sdk.memories.services import MemoryService

if TYPE_CHECKING:
    from django_ai_sdk.memories.models import EntryDocument
    from django_ai_sdk.types import UserType

# Only raster images render inline; anything else (html, svg, pdf...) downloads,
# so an uploaded file can never run as a page on our origin.
INLINE_FILE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


async def athread_file_response(thread_id: str, doc_id: str, *, user: UserType) -> FileResponse:
    """A thread file's bytes, for a user allowed to view the thread's files.

    Raises:
        NotFound: No such thread or file.
        PermissionDenied: `user` may not view the thread's files.
    """
    doc = await MemoryService.get_thread_file(thread_id, doc_id, user=user)
    return await sync_to_async(_file_response)(doc)


async def asource_file_response(doc_id: str, *, user: UserType) -> FileResponse:
    """A cited knowledge-base document, the original file. A PDF opens inline in the
    browser's own PDF viewer, so a link with ``#page=N`` opens it on that page;
    anything else is served like a thread file (see ``_file_response``).

    The PDF is the one exception to INLINE_FILE_TYPES: served as ``application/pdf``
    with ``nosniff``, so the browser's PDF viewer shows it and never renders it as
    a page. No ``sandbox`` CSP: it stops that viewer.

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
    """Page ``page`` of a knowledge-base document as a PNG, with the cited passages
    ``quotes`` marked: the real page, on any screen (phones do not show a PDF in a
    page). ``X-Mark`` says where the marks are, "top,left,bottom,right" as
    fractions of the page ("" when nothing is marked), to show the page there.

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


# For sync views (e.g. Django REST framework).
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
