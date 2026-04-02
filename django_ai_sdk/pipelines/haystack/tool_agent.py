"""
Haystack ToolAgent implementation.
"""

from typing import Any

from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.generators.chat import OpenAIChatGenerator
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
        generator: OpenAIChatGenerator,
    ) -> None:
        self.config = config
        self.generator = generator
        tool_names = [getattr(t, "name", type(t).__name__) for t in config.tools]
        logger.debug(
            f"[RAG_SERIALIZE] ToolAgent initialized with {len(config.tools)} tools: {tool_names}"
        )
        for i, tool in enumerate(config.tools):
            tool_type = type(tool).__name__
            logger.debug(f"[RAG_SERIALIZE] Tool #{i}: {tool_type}")

    def pipeline(self) -> Pipeline:
        """Build and return the fully connected pipeline."""
        logger.debug("[RAG_SERIALIZE] Building ToolAgent pipeline")

        agent = HaystackAgent(
            chat_generator=self.generator,
            tools=self.config.tools,
            system_prompt=self.config.system_prompt,
            exit_conditions=["text"],
        )

        pipeline = Pipeline()
        pipeline.add_component("agent", agent)

        logger.debug("[RAG_SERIALIZE] ToolAgent pipeline built successfully")
        return pipeline
