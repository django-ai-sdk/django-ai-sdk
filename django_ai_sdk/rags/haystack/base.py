from abc import ABC, abstractmethod
from typing import Any

from haystack import Pipeline
from haystack.tools import ComponentTool
from pydantic import BaseModel, Field

# TODO: move into prompts.py file, this should make maintenance easier.
DEFAULT_EXPANDER_PROMPT = """
You are a search query expansion assistant.

Your task is to generate {{n_expansions}} alternative search queries based on the user's original query.

The goal is to improve search recall by capturing different ways users might phrase the same question.

RULES:
1. Generate exactly {{n_expansions}} alternative queries
2. Each alternative should focus on different aspects or use different terminology
3. Use the SAME LANGUAGE as the original query
4. Output ONLY the alternative queries, one per line
5. Do NOT include the original query in your output
6. Make queries natural and conversational

Original query: {{query}}

Generate {{n_expansions}} alternative queries in the SAME language as the original:
"""


class BaseHaystackRAGConfig(BaseModel):
    """
    Base configuration for Haystack RAG implementations.

    This provides common configuration options for all Haystack-based RAG
    implementations, including query expansion settings.

    Attributes:
        top_k: Maximum number of documents to retrieve per query
        n_expansions: Number of query variations to generate (1 = no expansion)
        expander_model: LLM model to use for query expansion
        expander_prompt: Prompt template for query expansion
    """

    top_k: int = Field(default=5, ge=1, description="Maximum documents to retrieve per query")
    n_expansions: int = Field(
        default=4,
        ge=1,
        description="Number of query variations to generate (1 = no expansion)",
    )
    expander_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for query expansion",
    )
    expander_prompt: str = Field(
        default=DEFAULT_EXPANDER_PROMPT,
        description="Prompt template for query expansion",
    )


class HaystackRAGBase(ABC):
    """Abstract base class for Haystack RAG implementations."""

    _is_warmed_up: bool = False
    config: BaseHaystackRAGConfig

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the RAG by building the indexed document store (expensive).

        After warmup, subsequent build_pipeline() calls will use the cached store.
        """
        pass

    @property
    def needs_warmup(self) -> bool:
        """Check if warmup is needed."""
        return not self._is_warmed_up

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Build and return the RAG pipeline (query side, cheap).

        Returns:
            A Haystack Pipeline configured for RAG.
        """
        pass

    @abstractmethod
    def as_tool(self) -> ComponentTool:
        """
        Return the RAG pipeline wrapped as a ComponentTool.

        Returns:
            A ComponentTool wrapping the RAG pipeline.
        """
        pass
