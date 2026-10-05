from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.agents.base import Agent

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class Command(BaseCommand):
    help = "Rebuild the search index of every memory from scratch."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--memory", type=str, help="Only reindex this memory ID")
        parser.add_argument(
            "--agent",
            type=str,
            help="Agent whose RAG config builds the indexes; default: the first",
        )

    def handle(self, *args: str, **options: object) -> str | None:
        from django_ai_sdk.memories.models import Memory

        memories = Memory.objects.all().only("id", "name")
        if memory_id := cast("str | None", options.get("memory")):
            memories = memories.filter(id=memory_id)
            if not memories.exists():
                raise CommandError(f"Memory '{memory_id}' not found")
        targets = [(str(m.id), m.name) for m in memories]
        asyncio.run(self._reindex(targets, cast("str | None", options.get("agent"))))

    async def _reindex(self, memories: list[tuple[str, str]], agent_name: str | None) -> None:
        agent = await self._agent(agent_name)
        name = agent.__class__.__name__
        self.stdout.write(f"Reindexing {len(memories)} memories with {name}'s RAG config...")
        failed = 0
        for memory_id, memory_name in memories:
            try:
                rag = await agent.rag_provider.reindex(agent, memory_id, force_rebuild=True)
            except Exception as e:
                failed += 1
                logger.exception("Reindexing memory {} failed", memory_id)
                self.stdout.write(self.style.ERROR(f"  {memory_name} ({memory_id[:8]}): {e}"))
                continue
            note = "indexed" if rag is not None else "no documents, skipped"
            self.stdout.write(f"  {memory_name} ({memory_id[:8]}): {note}")
        if failed:
            raise CommandError(f"{failed} of {len(memories)} memories failed")
        self.stdout.write(self.style.SUCCESS("Reindex complete."))

    async def _agent(self, name: str | None) -> Agent:
        from django_ai_sdk.agents.registry import registry
        from django_ai_sdk.agents.services import AgentService

        try:
            registry.all()
        except RuntimeError:
            registry.setup(instantiate=True)
        agents = await AgentService.get_rag_agents()
        if name:
            agents = [a for a in agents if a.__class__.__name__ == name]
        if not agents:
            raise CommandError(f"No agent with a RAG provider{f' named {name!r}' if name else ''}")
        return agents[0]
