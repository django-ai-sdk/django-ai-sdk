from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class Command(BaseCommand):
    help = "Pre-warm RAG indexes for all assistants and memories."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--assistant",
            type=str,
            help="Only warm up a specific assistant (class name)",
        )
        parser.add_argument(
            "--memory",
            type=str,
            help="Only warm up a specific memory ID",
        )
        parser.add_argument(
            "--force-rebuild",
            action="store_true",
            help="Rebuild indexes from scratch, deleting existing ones",
        )

    def handle(self, *args: str, **options: object) -> str | None:
        from django_ai_sdk.memories.models import Memory

        memory_filter = cast("str | None", options.get("memory"))
        all_memory_ids = [str(m.id) for m in Memory.objects.all().only("id")]

        if memory_filter:
            if memory_filter not in all_memory_ids:
                raise CommandError(f"Memory '{memory_filter}' not found")
            memory_ids: list[str] = [memory_filter]
        else:
            memory_ids = all_memory_ids

        asyncio.run(
            self._warmup(
                memory_ids=memory_ids,
                assistant_filter=cast("str | None", options.get("assistant")),
                force_rebuild=bool(options.get("force_rebuild", False)),
            )
        )

    async def _warmup(
        self,
        memory_ids: list[str],
        assistant_filter: str | None,
        force_rebuild: bool,
    ) -> None:
        from django_ai_sdk.assistants.registry import registry

        try:
            assistants = registry.all()
        except RuntimeError:
            registry.setup(instantiate=True)
            assistants = registry.all()

        if assistant_filter:
            filtered = {
                aid: inst
                for aid, inst in assistants.items()
                if inst.__class__.__name__ == assistant_filter
            }
            if not filtered:
                raise CommandError(f"Assistant '{assistant_filter}' not found in registry")
            assistants = filtered

        if not assistants:
            self.stdout.write(self.style.WARNING("No assistants registered."))
            return

        if not memory_ids:
            self.stdout.write(self.style.WARNING("No memories found."))
            return

        for aid, inst in assistants.items():
            name = inst.__class__.__name__
            if inst.rag_provider is None:
                self.stdout.write(
                    self.style.WARNING(f"[{name}] No RAG provider configured, skipping")
                )
                continue

            self.stdout.write(f"[{name}] Warming up {len(memory_ids)} memory/memories...")

            for mid in memory_ids:
                try:
                    await inst.rag_provider.warmup(inst, mid, force_rebuild=force_rebuild)
                    self.stdout.write(f"  [{name}] Warmed memory {mid[:8]}")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  [{name}] memory {mid[:8]} FAILED: {e}"))
                    logger.exception("Warmup failed for %s memory %s", name, mid)

        self.stdout.write(self.style.SUCCESS("Warmup complete."))
