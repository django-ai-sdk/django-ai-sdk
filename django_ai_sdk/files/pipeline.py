from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.agent import Agent
    from django_ai_sdk.files.processors import FileProcessor
    from django_ai_sdk.files.transforms import BaseTransform

OnStep = Callable[[str | None], Awaitable[None]]


@dataclass
class PipelineResult:
    content: str
    data: dict | list = field(default_factory=dict)


def parse_data(data: Any) -> dict | list:
    """Normalize transform output for storage"""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, (dict, list)):
        return data
    return {}


class FilePipeline:
    """Composable file processing pipeline.

    Selects a file processor and chains transforms in sequence.
    Each agent can declare its own pipelines; the first one
    whose processor accepts the uploaded file is used.

    Example::

        FilePipeline(
            CSVFileProcessor(),
            transforms=[CSVTransform(), LLMExtractTransform(response_format=MySchema)],
        )
    """

    def __init__(
        self,
        file_processor: FileProcessor,
        transforms: list[BaseTransform] | None = None,
    ) -> None:
        # SAFEGUARD: Accept both a class and an instance
        self.file_processor = (
            file_processor() if isinstance(file_processor, type) else file_processor
        )
        self.transforms: list[BaseTransform] = transforms or []

    async def accepts(self, file: Any) -> bool:
        return await self.file_processor.is_valid(file)

    async def run(
        self,
        file: Any,
        *,
        agent: Agent | None = None,
        on_step: OnStep | None = None,
    ) -> PipelineResult | None:
        """Run processor then all transforms in sequence.
        Processor runs in a thread to avoid blocking the event loop.

        ``on_step``, if given, is awaited before the processor and before each
        transform, passing that component's ``step`` name (or ``None``). It's
        the pipeline's only checkpoint, so callers can also use it to raise
        and cancel the run, whether or not that boundary has a step name.
        """
        if not await self.accepts(file):
            return None

        if on_step:
            await on_step(self.file_processor.step)
        data: Any = await self.file_processor.run(file)
        if data is None:
            return None

        content = data

        for transform in self.transforms:
            if on_step:
                await on_step(transform.step)
            data = await transform.run(data, agent=agent)

        return PipelineResult(content=content, data=parse_data(data))
