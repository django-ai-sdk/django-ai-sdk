from __future__ import annotations

from asgiref.sync import async_to_sync
from django_tasks import task


@task(queue_name="default")
def process_document_upload(
    entry_doc_id: str,
    memory_id: str,
    assistant_id: str | None = None,
) -> None:
    """Sync task entry point for background file pipeline processing."""
    async_to_sync(run_file_pipeline)(entry_doc_id, memory_id, assistant_id)


async def run_file_pipeline(
    entry_doc_id: str,
    memory_id: str,
    assistant_id: str | None,
) -> None:
    from django_ai_sdk.assistants.services import AssistantService
    from django_ai_sdk.files.common import get_default_file_pipeline
    from django_ai_sdk.memories.models import Entry, EntryDocument, Memory

    entry_doc = await EntryDocument.objects.aget(id=entry_doc_id)
    memory = await Memory.objects.aget(id=memory_id)

    entry_doc.processing_status = EntryDocument.ProcessingStatus.PROCESSING
    await entry_doc.asave(update_fields=["processing_status", "updated_at"])

    try:
        if assistant_id:
            assistant = await AssistantService.get(assistant_id)
            pipeline = await assistant.get_file_pipeline(
                entry_doc.file
            ) or await get_default_file_pipeline(entry_doc.file)
        else:
            pipeline = await get_default_file_pipeline(entry_doc.file)

        result = await pipeline.run(entry_doc.file)

        if result is None:
            entry_doc.processing_status = EntryDocument.ProcessingStatus.FAILED
            entry_doc.processing_error = f"Unsupported or empty file: {entry_doc.file_name}"
            await entry_doc.asave(
                update_fields=["processing_status", "processing_error", "updated_at"]
            )
            return

        entry = await Entry.objects.acreate(
            memory=memory,
            name=entry_doc.file_name,
            content=result.content,
            data=result.data,
        )
        entry_doc.entry = entry
        entry_doc.extracted = bool(result.data)
        entry_doc.processing_status = EntryDocument.ProcessingStatus.COMPLETED
        await entry_doc.asave(
            update_fields=["entry", "extracted", "processing_status", "updated_at"]
        )
        # Entry post_save signal fires RAG indexing automatically

    except Exception as exc:
        entry_doc.processing_status = EntryDocument.ProcessingStatus.FAILED
        entry_doc.processing_error = str(exc)
        await entry_doc.asave(update_fields=["processing_status", "processing_error", "updated_at"])
        raise
