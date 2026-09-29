"""Serving a thread file's bytes, shared by the contrib Ninja and DRF layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.http import FileResponse

if TYPE_CHECKING:
    from django_ai_sdk.memories.models import EntryDocument

# Only raster images render inline; anything else (html, svg, pdf...) downloads,
# so an uploaded file can never run as a page on our origin.
INLINE_FILE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def thread_file_response(doc: EntryDocument) -> FileResponse:
    """Serve a thread file's bytes; call only after MemoryService.get_thread_file."""
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
