"""Serving a thread file's bytes, shared by the contrib Ninja and DRF layers.

The helpers take ids and a user, not a document: the lookup checks ``VIEW_FILE`` (see
``MemoryService.get_thread_file``), so there is no way to serve a file unchecked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync, sync_to_async
from django.http import FileResponse

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


# For sync views (e.g. Django REST framework).
thread_file_response = async_to_sync(athread_file_response)


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
