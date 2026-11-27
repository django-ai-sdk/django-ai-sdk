from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack import Document as HaystackDocument

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


def to_document(doc: RagDocument) -> HaystackDocument:
    return HaystackDocument(
        id=doc.id,
        content=doc.content,
        meta=doc.metadata,
    )


async def queryset_to_rag_documents(queryset: Any, **kwargs: Any) -> list[RagDocument]:
    """Each item in the queryset must implement ``to_rag_document() -> RagDocument``.

    Args:
        **kwargs: Ignored (backward-compat for memory_id callers)
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
