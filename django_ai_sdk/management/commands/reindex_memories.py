from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.memories.models import Memory

logger = get_logger(__name__)


def get_index_signature(rag: Any) -> dict[str, Any]:
    """What an index is built with: rebuild it when any of this changes."""
    config = getattr(rag, "config", None)
    return {
        "rag": type(rag).__name__,
        "dense_model": getattr(config, "dense_embedder_model", None)
        or getattr(config, "embedder_model", None),
        "sparse_model": getattr(config, "sparse_embedder_model", None),
        "dim": getattr(config, "embedding_dim", None),
    }


def get_stored_signature(memory: Memory) -> dict[str, Any]:
    """The signature of the last successful run, without its bookkeeping; {} if none."""
    stored = dict((memory.metadata or {}).get("sdk", {}).get("rag", {}))
    stored.pop("indexed_at", None)
    stored.pop("adopted", None)
    return stored


def get_store_location(rag: Any) -> str:
    """Where the index lives"""
    storage: Any = getattr(getattr(rag, "config", None), "storage", None)
    if getattr(storage, "is_server", False):
        return f"remote, {storage.location}"
    if getattr(storage, "is_persistent", False):
        return f"local, {Path(storage.persist_path).parent}"
    return "in memory, gone when this command exits"


def get_embeddings() -> str:
    """Get dense embeddings model and dimension"""
    if model := resolve_setting("AI_SDK_EMBEDDINGS_MODEL"):
        dim = resolve_setting("AI_SDK_EMBEDDINGS_DIM", 384)
        url = resolve_setting("OPENAI_API_URL") or ""
        return f"remote, {model} ({dim} dims) via {url}"
    return "local FastEmbed model (384 dims)"


async def store_signature(memory: Memory, signature: dict[str, Any], **extra: Any) -> None:
    """Record what the index is built with"""
    from django_ai_sdk.memories.models import Memory

    # reread, the metadata may have changed while this memory was indexing.
    rows = Memory.objects.filter(id=memory.id)
    metadata = await rows.values_list("metadata", flat=True).afirst() or {}
    sdk = metadata.get("sdk", {})
    sdk["rag"] = signature | extra | {"indexed_at": timezone.now().isoformat()}
    await rows.aupdate(metadata=metadata | {"sdk": sdk})


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
        """Rebuild the index from scratch unless it is up to date."""
        signature = get_index_signature(rag)
        if not force and get_stored_signature(memory) == signature:
            return "up to date, skipped"
        await rag.warmup(force_rebuild=True)
        # A failed embedding batch is only logged: fill it in, else don't record the index.
        if hasattr(rag, "embed_missing") and (left := (await rag.embed_missing())[1]):
            raise RuntimeError(f"{left} chunks without a dense vector")
        await store_signature(memory, signature)
        return "indexed"

    async def _fill_missing(self, memory: Memory, rag: Any, force: bool) -> str:
        """Embed only the chunks without a dense vector; never rebuild."""
        signature, stored = get_index_signature(rag), get_stored_signature(memory)
        if stored and stored != signature:
            # Vectors of another model can't be mixed into this index.
            was = ", ".join(f"{key}={value}" for key, value in stored.items())
            return self.style.WARNING(
                f"indexed with {was}, not the current model; rebuild it without --missing. Skipped"
            )
        if not hasattr(rag, "embed_missing"):
            return f"{type(rag).__name__} can't embed missing chunks, skipped"
        try:
            embedded, left = await rag.embed_missing()
        except ValueError as e:  # the store refuses a collection of another vector size
            raise RuntimeError(f"{e}; rebuild it without --missing") from e
        if left:
            raise RuntimeError(f"{embedded} chunks embedded, {left} still without a dense vector")
        note = f"{embedded} chunks embedded" if embedded else "complete"
        if not stored:  # indexed before signatures were stored
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
