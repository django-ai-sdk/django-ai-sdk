from django.conf import settings
from django.db.models import QuerySet
from django.utils import timezone
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.haystack import HaystackAdapter
from django_ai_sdk.assistants import auto_register
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.rags.config import VectorDBStorageConfig
from django_ai_sdk.rags.haystack import (
    BM25QueryExpanderRAG,
    BM25QueryExpanderRAGConfig,
    ChromaDBQueryExpanderRAG,
    ChromaDBQueryExpanderRAGConfig,
    QdrantBM25HybridRAG,
    QdrantBM25HybridRAGConfig,
)
from django_ai_sdk.rags.haystack.provider import HaystackRAGProvider
from django_ai_sdk.silos.models import Document, Silo, ThreadSilo
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.tracking.utils import track_embedding, track_llm
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import ComponentTool, Tool
from haystack.utils import Secret


def get_datetime() -> dict:
    """Get current time and date in Europe/Amsterdam timezone."""
    tz = timezone.get_current_timezone()
    nowtz = timezone.now().astimezone(tz)

    return {
        "today": nowtz.date().isoformat(),
        "current_time": nowtz.timetz().isoformat(),
    }


def get_today() -> Tool:
    """Current date and time tool."""
    return Tool(
        name="get_today",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )


@auto_register
class PirateBasicAssistant(Assistant):
    name = "Basic Pirate Assistant"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = [
        "You are a helpful AI assistant who always responds like a pirate.Use pirate language, expressions, and mannerisms in all your responses.",
        "Be creative with pirate slang but keep responses helpful and informative.",
        "Make sure to generate pirate speech in the same language as the user's query.",
        "When interacting with tools, use the response to answer the user's query, but keep the pirate tone and style in your response.",
        "Do not repeat the tool response as is, but use it to answer the user's query in a pirate style.",
        "Do not mix languages, if the user's question is in English, respond in English pirate style, if the user's question is in Dutch, respond in Dutch pirate style.",
        "when calling a tool and the response is empty or has a error, please do not make up a response, but instead respond with 'Arrr, I couldn't find any treasure on that one!' or 'Arrr, there be an error with that tool!' in the appropriate pirate language.",
        "When context is added, please use it to generate a answer that answers the user question, but do not make up a response if the context is empty or does not contain relevant information, instead respond with 'Arrr, I couldn't find any treasure on that one!' in the appropriate pirate language.",
        "Parts of the question may or may-not be answerable, then make sure to only respond on those parts with: 'Arrr, I couldn't find any treasure on that one!'"
        "If no information is available, then respond with general knowledge in pirate style, but make sure to include the phrase 'Arrr, I couldn't find any treasure on that one!' in your response to indicate that you don't have specific information to answer the question. But respond with general knowledge.",
    ]

    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter

    # Use the new RAG provider pattern for Haystack
    rag_provider = HaystackRAGProvider()

    async def get_rag_queryset(self, silo_id: str | None = None) -> QuerySet[Document]:
        """Return queryset of documents for RAG."""
        if silo_id:
            return Document.objects.filter(silo_id=silo_id)
        return Document.objects.all()

    async def get_rag_pipeline_bm25(
        self, silo_id: str | None = None
    ) -> BM25QueryExpanderRAG | None:
        """Build BM25 RAG pipeline for document retrieval."""
        documents = await self.get_rag_documents(silo_id)
        if not documents:
            return None

        return BM25QueryExpanderRAG(
            documents=documents,
            config=BM25QueryExpanderRAGConfig(
                top_k=5,
                n_expansions=4,
            ),
        )

    async def get_rag_pipeline_chromadb(
        self, silo_id: str | None = None
    ) -> ChromaDBQueryExpanderRAG | None:
        """Build ChromaDB RAG pipeline for document retrieval."""
        documents = await self.get_rag_documents(silo_id)
        if not documents:
            return None

        return ChromaDBQueryExpanderRAG(
            documents=documents,
            config=ChromaDBQueryExpanderRAGConfig(
                top_k=5,
                n_expansions=4,
                expander_model="openai/gpt-oss-120b",
            ),
        )

    async def get_rag_pipeline_qdrant(
        self, silo_id: str | None = None
    ) -> QdrantBM25HybridRAG | None:
        """Build Qdrant Hybrid RAG pipeline (SPLADE + BGE embeddings with RRF)."""
        documents = await self.get_rag_documents(silo_id)
        if not documents:
            return None

        return QdrantBM25HybridRAG(
            documents=documents,
            config=QdrantBM25HybridRAGConfig(
                top_k=5,
                n_expansions=4,
                expander_model="openai/gpt-oss-120b",
                meta_fields_to_embed=["file_name", "keywords", "facts"],
                storage=VectorDBStorageConfig.from_settings(silo_id),
            ),
        )

    @track_llm
    @track_embedding
    async def get_rag_pipeline(self, silo_id: str | None = None) -> QdrantBM25HybridRAG | None:
        """Build RAG pipeline for document retrieval."""
        return await self.get_rag_pipeline_qdrant(silo_id)

    async def get_pipeline_adapter(self, thread_id: str | None = None) -> HaystackAdapter:
        """Create Haystack pipeline adapter with multi-silo RAG tools."""

        # Get storage adapter
        storage_adapter = await self.get_storage_adapter(thread_id)

        # Build generator
        generator = OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

        # Create tools list
        tools = [get_today()]

        # Add RAG tools for each silo
        if self.rag_provider and thread_id:
            silo_links = ThreadSilo.objects.filter(thread_id=thread_id).prefetch_related("silo")

            async for link in silo_links:
                try:
                    silo = await Silo.objects.aget(id=link.silo.id)
                    spec = await silo.get_tool_spec()

                    rag = await self.rag_provider.get_rag_instance(self, str(silo.id))
                    if rag:
                        tool = rag.get_tool(spec)
                        tools.append(tool)

                except Silo.DoesNotExist:
                    continue

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

        return HaystackAdapter(
            pipeline=pipeline,
            generator_component=generator,
            storage_adapter=storage_adapter,
        )
