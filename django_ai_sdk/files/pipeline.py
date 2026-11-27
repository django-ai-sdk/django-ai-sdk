from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.files.processors import FileProcessor
    from django_ai_sdk.files.transforms import BaseTransform

OnStep = Callable[[str | None], Awaitable[None]]


@dataclass
class PipelineResult:
    content: str
    data: dict | list = field(default_factory=dict)


def parse_data(data: Any) -> dict | list:
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, (dict, list)):
        return data
    return {}


class FilePipeline:
    def __init__(
        self,
        file_processor: FileProcessor,
        transforms: list[BaseTransform] | None = None,
    ) -> None:
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
        """``on_step``, if given, is awaited before the processor and before each
        transform, passing that component's ``step`` name (or ``None``).
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
