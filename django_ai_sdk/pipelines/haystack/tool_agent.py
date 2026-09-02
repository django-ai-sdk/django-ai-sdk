"""
Haystack ToolAgent implementation.
"""

from __future__ import annotations

from typing import Any

from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class ToolAgentConfig(BaseModel):
    """Configuration for ToolAgent."""

    model_config = {"arbitrary_types_allowed": True}

    model: str
    system_prompt: str
    tools: list[Any] = []


class ToolAgent:
    """ToolAgent that wraps a Haystack Agent with tools."""

    def __init__(
        self,
        config: ToolAgentConfig,
        generator: Any,
    ) -> None:
        self.config = config
        self.generator = generator
        logger.debug(f"ToolAgent initialized with {len(config.tools)} tools")

    def pipeline(self) -> Pipeline:
        """Build and return the fully connected pipeline."""
        logger.debug("Building ToolAgent pipeline")

        agent = HaystackAgent(
            chat_generator=self.generator,
            tools=self.config.tools,
            system_prompt=self.config.system_prompt,
            exit_conditions=["text"],
        )

        pipeline = Pipeline()
        pipeline.add_component("agent", agent)

        logger.debug("ToolAgent pipeline built successfully")
        return pipeline
