from typing import Any

from haystack import Document as HaystackDocument

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


def to_document(doc: RagDocument) -> HaystackDocument:
    """
    Convert a RagDocument to a Haystack Document.

    Args:
        doc: RagDocument to convert

    Returns:
        Haystack Document

    Example:
        haystack_doc = to_document(rag_doc)
    """
    return HaystackDocument(
        id=doc.id,
        content=doc.content,
        meta=doc.metadata,
    )


async def queryset_to_rag_documents(queryset: Any, **kwargs: Any) -> list[RagDocument]:
    """
    Convert any Django QuerySet to a list of RagDocuments.

    Each item in the queryset must implement ``to_rag_document() -> RagDocument``.
    Works with Entry, Document (alias), or any proxy subclass (NoteEntry, etc.).

    The ``memory_id`` keyword arg is accepted for backward compatibility with
    callers that still pass it — it is not used here because each model's own
    ``to_rag_document()`` knows how to populate metadata.

    Args:
        queryset: Django QuerySet whose items implement to_rag_document()
        **kwargs: Ignored (backward-compat for memory_id callers)

    Returns:
        List of RagDocuments (vendor-neutral)

    Example:
        documents = await queryset_to_rag_documents(Entry.objects.filter(memory_id=...))
    """
    rag_docs = []
    skipped_count = 0
    async for obj in queryset:
        try:
            doc = obj.to_rag_document()
            if doc.content.strip():
                rag_docs.append(doc)
            else:
                skipped_count += 1
        except (AttributeError, NotImplementedError):
            skipped_count += 1

    logger.debug(
        f"Converted queryset to {len(rag_docs)} RagDocuments (skipped {skipped_count} empty/incompatible)"
    )
    return rag_docs
