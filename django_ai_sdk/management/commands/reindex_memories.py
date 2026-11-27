from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.config import DEFAULT_EMBEDDING_DIM
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.memories.models import Memory

logger = get_logger(__name__)


def get_index_signature(rag: Any) -> dict[str, Any]:
    config = getattr(rag, "config", None)
    return {
        "rag": type(rag).__name__,
        "dense_model": getattr(config, "dense_embedder_model", None)
        or getattr(config, "embedder_model", None),
        "sparse_model": getattr(config, "sparse_embedder_model", None),
        "dim": getattr(config, "embedding_dim", None),
    }


def get_stored_signature(memory: Memory) -> dict[str, Any]:
    stored = dict((memory.metadata or {}).get("sdk", {}).get("rag", {}))
    stored.pop("indexed_at", None)
    stored.pop("adopted", None)
    return stored


def get_store_location(rag: Any) -> str:
    storage: Any = getattr(getattr(rag, "config", None), "storage", None)
    if getattr(storage, "is_server", False):
        return f"remote, {storage.location}"
    if getattr(storage, "is_persistent", False):
        return f"local, {Path(storage.persist_path).parent}"
    return "in memory, gone when this command exits"


def get_embeddings() -> str:
    if model := resolve_setting("AI_SDK_EMBEDDINGS_MODEL"):
        dim = resolve_setting("AI_SDK_EMBEDDINGS_DIM", DEFAULT_EMBEDDING_DIM)
        url = resolve_setting("OPENAI_API_URL") or ""
        return f"remote, {model} ({dim} dims) via {url}"
    return f"local FastEmbed model ({DEFAULT_EMBEDDING_DIM} dims)"


async def store_signature(memory: Memory, signature: dict[str, Any], **extra: Any) -> None:
    await sync_to_async(write_signature)(memory.id, signature | extra)


def write_signature(memory_id: Any, signature: dict[str, Any]) -> None:
    from django_ai_sdk.memories.models import Memory

    with transaction.atomic():
        rows = Memory.objects.select_for_update().filter(id=memory_id)
        metadata = rows.values_list("metadata", flat=True).first() or {}
        sdk = metadata.get("sdk", {})
        sdk["rag"] = signature | {"indexed_at": timezone.now().isoformat()}
        rows.update(metadata=metadata | {"sdk": sdk})


class Command(BaseCommand):
    help = (
        "Rebuild the search index of every memory built with another embedding model or "
        "dimension (--force: all); --missing only embeds chunks stored without a dense vector."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--memory", type=str, help="Only reindex this memory ID")
        parser.add_argument(
            "--agent",
            type=str,
            help="Agent whose RAG config builds the indexes; default: the first",
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--force", action="store_true", help="Rebuild every memory, also up-to-date ones"
        )
        mode.add_argument(
            "--missing",
            action="store_true",
            help="Rebuild nothing: only embed chunks stored without a dense vector",
        )

    def handle(self, *args: str, **options: Any) -> None:
        from django_ai_sdk.memories.models import Memory

        memories = Memory.objects.only("id", "name", "metadata")
        if memory_id := options["memory"]:
            memories = memories.filter(id=memory_id)
            if not memories.exists():
                raise CommandError(f"Memory '{memory_id}' not found")
        asyncio.run(self._run(list(memories), options))

    async def _run(self, memories: list[Memory], options: dict[str, Any]) -> None:
        agent = await self._agent(options["agent"])
        if options["missing"]:
            task, step = "Embedding chunks without a dense vector in", self._fill_missing
        else:
            task, step = (
                f"Reindexing {'all' if options['force'] else 'out-of-date'} of",
                self._rebuild,
            )
        self.stdout.write(
            f"{task} {len(memories)} memories with {type(agent).__name__}'s RAG config..."
        )
        self.stdout.write(f"Dense embeddings: {get_embeddings()}")

        location_shown, failed = False, 0
        for memory in memories:
            label = f"  {memory.name} ({str(memory.id)[:8]})"
            try:
                rag = await agent.get_rag_pipeline(str(memory.id))
                if rag is None:
                    self.stdout.write(f"{label}: no documents, skipped")
                    continue
                if not location_shown:
                    self.stdout.write(f"Vector store: {get_store_location(rag)}")
                    location_shown = True
                note = await step(memory, rag, force=options["force"])
            except Exception as e:
                failed += 1
                logger.exception("Reindexing memory {} failed", memory.id)
                self.stdout.write(self.style.ERROR(f"{label}: {e}"))
                continue
            self.stdout.write(f"{label}: {note}")

        if failed:
            raise CommandError(f"{failed} of {len(memories)} memories failed")
        self.stdout.write(self.style.SUCCESS("Reindex complete."))

    async def _rebuild(self, memory: Memory, rag: Any, force: bool) -> str:
        signature = get_index_signature(rag)
        if not force and get_stored_signature(memory) == signature:
            return "up to date, skipped"
        await rag.warmup(force_rebuild=True)
        if hasattr(rag, "embed_missing") and (left := (await rag.embed_missing())[1]):
            raise RuntimeError(f"{left} chunks without a dense vector")
        await store_signature(memory, signature)
        return "indexed"

    async def _fill_missing(self, memory: Memory, rag: Any, force: bool) -> str:
        signature, stored = get_index_signature(rag), get_stored_signature(memory)
        if stored and stored != signature:
            was = ", ".join(f"{key}={value}" for key, value in stored.items())
            return self.style.WARNING(
                f"indexed with {was}, not the current model; rebuild it without --missing. Skipped"
            )
        if not hasattr(rag, "embed_missing"):
            return f"{type(rag).__name__} can't embed missing chunks, skipped"
        try:
            embedded, left = await rag.embed_missing()
        except ValueError as e:
            raise RuntimeError(f"{e}; rebuild it without --missing") from e
        if left:
            raise RuntimeError(f"{embedded} chunks embedded, {left} still without a dense vector")
        note = f"{embedded} chunks embedded" if embedded else "complete"
        if not stored:
            await store_signature(memory, signature, adopted=True)
            note += " (no signature yet: dimension checked, model assumed current)"
        return note

    async def _agent(self, name: str | None) -> Agent:
        from django_ai_sdk.agents.registry import registry
        from django_ai_sdk.agents.services import AgentService

        try:
            registry.all()
        except RuntimeError:
            registry.setup(instantiate=True)
        agents = await AgentService.get_rag_agents()
        if name:
            agents = [a for a in agents if type(a).__name__ == name]
        if not agents:
            raise CommandError(f"No agent with a RAG provider{f' named {name!r}' if name else ''}")
        return agents[0]
