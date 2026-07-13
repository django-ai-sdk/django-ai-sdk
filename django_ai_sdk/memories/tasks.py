from __future__ import annotations

import asyncio
from typing import Any

from asgiref.sync import async_to_sync
from django.conf import settings
from django_tasks import task

# No pipeline step reports progress within this ceiling; a hung/leaked
# connection or a wedged upstream call fails the task instead of leaving the
# document stuck in PROCESSING forever (django_tasks has no native timeout).
PIPELINE_TIMEOUT_SECONDS = getattr(settings, "AI_SDK_FILE_PIPELINE_TIMEOUT", 900)


class _Cancelled(Exception):
    """Raised internally when a step boundary notices cancelled_at is set."""


@task(queue_name="default")
def process_document_upload(
    entry_doc_id: str,
    memory_id: str,
    assistant_id: str | None = None,
) -> dict[str, Any] | None:
    """Sync task entry point for background file pipeline processing.

    The return value becomes ``DBTaskResult.return_value``, retrievable via
    ``aget_task_status`` on success.
    """
    return async_to_sync(run_file_pipeline)(entry_doc_id, memory_id, assistant_id)


async def run_file_pipeline(
    entry_doc_id: str,
    memory_id: str,
    assistant_id: str | None,
) -> dict[str, Any] | None:
    from django.utils import timezone

    from django_ai_sdk.assistants.services import AssistantService
    from django_ai_sdk.files.common import get_default_file_pipeline
    from django_ai_sdk.memories.models import Entry, EntryDocument, Memory

    # The document may have been deleted (user cancelled an in-flight upload) or
    # its memory removed before the worker picked up the task — treat as a no-op.
    try:
        entry_doc = await EntryDocument.objects.aget(id=entry_doc_id)
        memory = await Memory.objects.aget(id=memory_id)
    except (EntryDocument.DoesNotExist, Memory.DoesNotExist):
        return

    # Only writes if still PROCESSING, so a concurrent cancel/finish wins
    # instead of being clobbered. Returns whether the write applied.
    async def _transition(*, status: str, error: str) -> bool:
        updated = await EntryDocument.objects.filter(
            id=entry_doc.id, processing_status=EntryDocument.ProcessingStatus.PROCESSING
        ).aupdate(
            processing_status=status,
            processing_error=error,
            processing_step=None,
            updated_at=timezone.now(),
        )
        return bool(updated)

    async def _mark_cancelled() -> None:
        await _transition(
            status=EntryDocument.ProcessingStatus.CANCELLED, error="Cancelled by user"
        )

    if entry_doc.cancelled_at is not None:
        # Cancelled while still queued
        return

    entry_doc.processing_status = EntryDocument.ProcessingStatus.PROCESSING
    await entry_doc.asave(update_fields=["processing_status", "updated_at"])

    async def _on_step(step: str | None) -> None:
        updated = await EntryDocument.objects.filter(
            id=entry_doc.id, cancelled_at__isnull=True
        ).aupdate(processing_step=step, updated_at=timezone.now())
        if not updated:
            raise _Cancelled

    try:
        if assistant_id:
            assistant = await AssistantService.get(assistant_id)
            pipeline = await assistant.get_file_pipeline(
                entry_doc.file
            ) or await get_default_file_pipeline(entry_doc.file)
        else:
            pipeline = await get_default_file_pipeline(entry_doc.file)

        result = await asyncio.wait_for(
            pipeline.run(entry_doc.file, on_step=_on_step),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )

        if result is None:
            await _transition(
                status=EntryDocument.ProcessingStatus.FAILED,
                error=f"Unsupported or empty file: {entry_doc.file_name}",
            )
            return

        entry = await Entry.objects.acreate(
            memory=memory,
            name=entry_doc.file_name,
            content=result.content,
            data=result.data,
        )
        extracted = bool(result.data)
        updated = await EntryDocument.objects.filter(
            id=entry_doc.id, processing_status=EntryDocument.ProcessingStatus.PROCESSING
        ).aupdate(
            entry=entry,
            extracted=extracted,
            processing_status=EntryDocument.ProcessingStatus.COMPLETED,
            processing_step=None,
            updated_at=timezone.now(),
        )
        if not updated:
            return  # cancelled after the last checkpoint; leave that status alone
        # Entry post_save signal fires RAG indexing automatically
        return {"entry_id": str(entry.id), "extracted": extracted}

    except _Cancelled:
        await _mark_cancelled()
        return
    except TimeoutError as exc:
        # asyncio.wait_for's own TimeoutError has no message; prefer that over
        # an inner one's message if there is one.
        if not await _transition(
            status=EntryDocument.ProcessingStatus.FAILED,
            error=str(exc) or f"Processing timed out after {PIPELINE_TIMEOUT_SECONDS} seconds",
        ):
            await _mark_cancelled()
        raise
    except Exception as exc:
        if not await _transition(status=EntryDocument.ProcessingStatus.FAILED, error=str(exc)):
            await _mark_cancelled()
        raise
