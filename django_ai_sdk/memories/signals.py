from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.models import Entry
from django_ai_sdk.rags.utils import queryset_to_rag_documents

if TYPE_CHECKING:
    from django.db.models.base import Model

logger = get_logger(__name__)


@receiver(post_save, sender=Entry)
async def on_entry_saved(
    sender: type[Model], instance: Entry, created: bool, **kwargs: object
) -> None:
    """Add or update document in RAG index when Entry is saved."""
    memory_id = str(instance.memory_id)
    logger.info(f"Entry saved for memory_id={memory_id}")

    for assistant_id, assistant in registry.all().items():
        provider = assistant.rag_provider
        if provider is None:
            continue

        queryset = Entry.objects.filter(id=instance.id)
        documents = await queryset_to_rag_documents(queryset)

        if documents:
            await provider.add_documents(assistant, memory_id, documents)
            logger.info(f"Added/updated document in RAG for {memory_id}")


@receiver(post_delete, sender=Entry)
async def on_entry_deleted(sender: type[Model], instance: Entry, **kwargs: object) -> None:
    """Remove document from RAG index when Entry is deleted."""
    memory_id = str(instance.memory_id)
    logger.info(f"Entry deleted for memory_id={memory_id}")

    for assistant_id, assistant in registry.all().items():
        provider = assistant.rag_provider
        if provider is None:
            continue

        await provider.remove_documents(assistant, memory_id, [str(instance.id)])
        logger.info(f"Removed document from RAG for {memory_id}")
