from typing import Any

from haystack import Document as HaystackDocument

from django_ai_sdk.rags.schemas import RagDocument


def rag_document_to_haystack(doc: RagDocument) -> HaystackDocument:
    """
    Convert a RagDocument to a Haystack Document.

    Args:
        doc: RagDocument to convert

    Returns:
        Haystack Document

    Example:
        haystack_doc = rag_document_to_haystack(rag_doc)
    """
    return HaystackDocument(
        id=doc.id,
        content=doc.content,
        meta=doc.metadata,
    )


# TODO: this might become extraction to rag documents, it is now our Document queryset
# But it now depends heavily on the Document model itself.
# A true queryset_to_rag_documents should be generic and work with any queryset
async def queryset_to_rag_documents(queryset: Any, silo_id: Any = None) -> list[RagDocument]:
    """
    Convert Django QuerySet to list of RagDocuments.
    This is the vendor-neutral way to get documents for RAG.
    The returned RagDocuments can be passed to any RAG adapter

    Args:
        queryset: Django QuerySet of Document objects (async iterable)
        silo_id: Optional silo_id for metadata

    Returns:
        List of RagDocuments with metadata:
        - file_name, silo_id, keywords, facts

    Example:
        documents = await queryset_to_rag_documents(queryset, silo_id)
    """
    from django_ai_sdk.silos.utils import get_prompt_metadata

    rag_docs = []
    async for doc in queryset:
        if not doc.content.strip():
            continue

        extraction = doc.extraction
        if extraction:
            combined_content = get_prompt_metadata(doc.content, extraction)
        else:
            combined_content = doc.content

        rag_docs.append(
            RagDocument(
                id=str(doc.id),
                content=combined_content,
                metadata={
                    "file_name": doc.file_name,
                    "silo_id": str(doc.silo_id) if doc.silo_id else silo_id or "",
                    "keywords": ". ".join(extraction.keywords) if extraction else "",
                    "facts": ". ".join(fact.text for fact in extraction.facts)
                    if extraction
                    else "",
                },
            )
        )

    return rag_docs
