from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import auto_register
from django_ai_sdk.citations import DefaultCitationFormatter
from django_ai_sdk.common import prompt
from django_ai_sdk.files import FilePipeline, TextFileProcessor
from django_ai_sdk.generators import openai_responses_chat
from django_ai_sdk.memories.models import Entry
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.rags import (
    BM25QueryExpanderRAG,
    BM25QueryExpanderRAGConfig,
    QdrantBM25HybridRAG,
    QdrantBM25HybridRAGConfig,
)
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.suggestions import DefaultSuggestionGenerator

from .extraction import PirateExtractionAgent
from .tools import get_memory_files, get_today
from .transforms import DocumentExtractionTransform

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.db.models import QuerySet


@auto_register
class PirateBasicAgent(Agent):
    name = "Basic Pirate Agent"
    model = settings.AI_SDK_DEFAULT_MODEL
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "low", "summary": "auto"}}
    instructions = prompt("""\
        You are a helpful AI agent who always responds like a pirate.
        Use pirate language, expressions, and mannerisms in all your responses.
        - Be creative with pirate slang but keep responses helpful and informative.
        - Make sure to generate pirate speech in the same language as the user's query.
        - When interacting with tools, use the response to answer the user's query, but keep the pirate tone and style in your response.
        - Do not repeat the tool response as is, but use it to answer the user's query in a pirate style.
        - Do not mix languages, if the user's question is in English, respond in English pirate style, if the user's question is in Dutch, respond in Dutch pirate style.
          when calling a tool and the response is empty or has a error, please do not make up a response, but instead respond with 'Arrr, I couldn't find any treasure on that one!' or 'Arrr, there be an error with that tool!' in the appropriate pirate language.
        - When context is added, please use it to generate a answer that answers the user question, but do not make up a response if the context is empty or does not contain relevant information, instead respond with 'Arrr, I couldn't find any treasure on that one!' in the appropriate pirate language.
        - Parts of the question may or may-not be answerable, then make sure to only respond on those parts with: 'Arrr, I couldn't find any treasure on that one!'
        - If no information is available, then respond with general knowledge in pirate style, but make sure to include the phrase 'Arrr, I couldn't find any treasure on that one!' in your response to indicate that you don't have specific information to answer the question. But respond with general knowledge.
    """)

    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    max_history = 20

    # Enable file upload UI for this agent
    file_upload = True

    file_pipelines = [
        FilePipeline(
            TextFileProcessor(),
            transforms=[
                DocumentExtractionTransform(PirateExtractionAgent()),
            ],
        ),
    ]

    # Use the new RAG provider pattern for Haystack
    rag_provider = RAGProvider()

    tools: list = [get_today, get_memory_files]

    # Registry keys of installed integration apps.
    integrations: list[str] = ["linear", "weather"]

    citation_formatter_class = DefaultCitationFormatter
    suggestion_generator = DefaultSuggestionGenerator

    async def get_rag_queryset(self, memory_id: str | None = None) -> QuerySet[Entry]:
        """Return queryset of documents for RAG."""
        if memory_id:
            return Entry.objects.filter(memory_id=memory_id)
        return Entry.objects.all()

    async def get_rag_pipeline_bm25(
        self, memory_id: str | None = None
    ) -> BM25QueryExpanderRAG | None:
        """Build BM25 RAG pipeline for document retrieval."""
        documents = await self.get_rag_documents(memory_id)
        if not documents:
            return None

        return BM25QueryExpanderRAG(
            documents=documents,
            config=BM25QueryExpanderRAGConfig(
                top_k=5,
                n_expansions=4,
            ),
        )

    async def get_rag_pipeline_qdrant(
        self, memory_id: str | None = None
    ) -> QdrantBM25HybridRAG | None:
        """Build Qdrant Hybrid RAG pipeline (SPLADE + BGE embeddings with RRF)."""
        documents = await self.get_rag_documents(memory_id)
        if not documents:
            return None

        return QdrantBM25HybridRAG(
            documents=documents,
            config=QdrantBM25HybridRAGConfig(
                top_k=5,
                n_expansions=4,
                expander_model="openai/gpt-oss-120b",
                meta_fields_to_embed=["file_name", "keywords", "facts"],
                storage=QdrantStorageConfig.from_settings(memory_id),
            ),
        )

    async def get_rag_pipeline(self, memory_id: str | None = None) -> QdrantBM25HybridRAG | None:
        """Build RAG pipeline for document retrieval."""
        return await self.get_rag_pipeline_qdrant(memory_id)

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        generator = self.get_llm()
        return Run(generator=generator)

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Stream:
        """Create Haystack pipeline adapter with multi-memory RAG tools."""

        generator = self.get_llm()

        # Get storage adapter
        storage_adapter = await self.get_storage_adapter(thread_id)

        # One registry/formatter per request: indices stay cumulative across
        # multiple RAG tool calls in a turn, and reset between turns.
        citation_registry = self.get_citation_registry()
        citation_formatter = self.get_citation_formatter()

        # Get tools
        tools = await self.get_tools(
            thread_id=thread_id or "",
            user=user,
        )

        # Add RAG tools for each active memory link
        if self.rag_provider and thread_id:
            rag_tools = await self.get_rag_tools(
                thread_id=thread_id,
                citation_registry=citation_registry,
                citation_formatter=citation_formatter,
                user=user,
            )
            tools.extend(rag_tools)

        # Build tool agent with all tools
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=tools,
            ),
            generator=generator,
        )

        pipeline = tool_agent.pipeline()

        return Stream(
            pipeline=pipeline,
            generator=generator,
            storage_adapter=storage_adapter,
            citation_registry=citation_registry,
            suggestion_generator=self.get_suggestion_generator(),
        )
