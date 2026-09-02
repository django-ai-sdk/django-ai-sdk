from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from django.test import override_settings
from django_ai_sdk.adapters.base import Run
from django_ai_sdk.adapters import base
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agent import Agent
from django_ai_sdk.generators import (
    anthropic_chat,
    azure_openai_chat,
    azure_openai_responses_chat,
    huggingface_api_chat,
    mistral_chat,
    ollama_chat,
    openai_chat,
    openai_responses_chat,
    openrouter_chat,
    schema_kwargs,
)
from haystack.components.generators.chat import OpenAIChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ReasoningContent, StreamingChunk
from pydantic import BaseModel


class Answer(BaseModel):
    """Schema used for the structured-output assertions."""

    text: str


class TestOpenAIFactories:
    """The two OpenAI factories and their settings wiring."""

    @override_settings(OPENAI_API_KEY=None, OPENAI_API_URL=None)
    def test_builds_without_settings(self):
        # Credentials are omitted when unset, so Haystack's own env-var default
        # applies instead of raising AttributeError.
        generator = openai_responses_chat()

        assert isinstance(generator, OpenAIResponsesChatGenerator)
        assert generator.api_base_url is None
        assert generator.model

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_API_URL="https://example.test/v1")
    def test_reads_vendor_settings(self):
        generator = openai_chat(model="gpt-4o-mini")

        assert generator.api_key.resolve_value() == "sk-test"
        assert generator.api_base_url == "https://example.test/v1"
        assert generator.model == "gpt-4o-mini"

    @override_settings(OPENAI_API_URL="https://example.test/v1")
    def test_explicit_kwargs_win(self):
        generator = openai_chat(api_base_url="https://override.test/v1")

        assert generator.api_base_url == "https://override.test/v1"

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_no_reasoning_by_default(self):
        # Reasoning is opt-in: an agent or caller passes it explicitly.
        assert openai_responses_chat().generation_kwargs == {}

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_reasoning_passes_through(self):
        reasoning = {"effort": "low", "summary": "auto"}
        generator = openai_responses_chat(generation_kwargs={"reasoning": reasoning})

        assert generator.generation_kwargs["reasoning"] == reasoning


def _deployment_of(generator):
    """The Azure classes disagree on whether the attribute is private."""
    return getattr(generator, "azure_deployment", None) or generator._azure_deployment


class TestAzureFactories:
    """Azure calls the model a deployment."""

    @override_settings(
        AZURE_OPENAI_API_KEY="az-test",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
        AZURE_OPENAI_API_VERSION="2024-12-01-preview",
    )
    def test_model_maps_to_deployment(self):
        generator = azure_openai_chat(model="my-deployment")

        assert _deployment_of(generator) == "my-deployment"

    @override_settings(
        AZURE_OPENAI_API_KEY="az-test",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
        AZURE_OPENAI_DEPLOYMENT="settings-deployment",
    )
    def test_deployment_falls_back_to_setting(self):
        assert _deployment_of(azure_openai_responses_chat()) == "settings-deployment"


class TestOptionalVendors:
    """Vendors that ship in their own pip package."""

    @pytest.mark.parametrize(
        ("factory", "class_name", "settings_overrides"),
        [
            (anthropic_chat, "AnthropicChatGenerator", {"ANTHROPIC_API_KEY": "an-test"}),
            (
                mistral_chat,
                "MistralChatGenerator",
                {"MISTRAL_API_KEY": "mi-test", "MISTRAL_API_URL": "https://example.test/v1"},
            ),
            (ollama_chat, "OllamaChatGenerator", {"OLLAMA_API_URL": "http://ollama.test:11434"}),
            (
                openrouter_chat,
                "OpenRouterChatGenerator",
                {"OPENROUTER_API_KEY": "or-test"},
            ),
        ],
    )
    def test_factory_builds_its_generator(self, factory, class_name, settings_overrides):
        # Every kwarg the factory sets must exist on the real class - Anthropic,
        # for one, has no base-URL parameter at all.
        with override_settings(**settings_overrides):
            generator = factory(model="test-model")

        assert type(generator).__name__ == class_name

    @override_settings(
        HUGGINGFACE_API_KEY="hf-test",
        HUGGINGFACE_API_URL="https://endpoint.test",
    )
    def test_huggingface_uses_the_endpoint_url(self):
        # Serverless inference validates the model against the Hub over the
        # network at construction; a dedicated endpoint URL does not.
        generator = huggingface_api_chat()

        assert type(generator).__name__ == "HuggingFaceAPIChatGenerator"
        assert generator.api_params == {"url": "https://endpoint.test"}

    @override_settings(OLLAMA_API_URL="http://ollama.test:11434")
    def test_ollama_reasoning_moves_to_the_init_param(self):
        # `think` is an init param on OllamaChatGenerator, but agents declare
        # reasoning uniformly through llm_kwargs -> generation_kwargs.
        generator = ollama_chat(model="qwen3:0.6b", generation_kwargs={"think": True})

        assert generator.think is True
        assert "think" not in (generator.generation_kwargs or {})

    def test_importing_one_vendor_does_not_import_the_others(self):
        # Each vendor ships in its own package.
        for module in [m for m in sys.modules if m.startswith("django_ai_sdk.generators.")]:
            del sys.modules[module]

        importlib.import_module("django_ai_sdk.generators").openai_chat

        loaded = {m.rsplit(".", 1)[-1] for m in sys.modules if m.startswith("django_ai_sdk.generators.")}
        assert not loaded & {"anthropic", "mistral", "ollama", "openrouter", "huggingface", "transformers"}


class TestStructuredOutput:
    """`Run` must send the kwarg the generator's API actually accepts.

    Sending the other one raises TypeError from the OpenAI client before any
    request is made, so this is not a recoverable mistake.
    """

    @staticmethod
    def _capture(generator):
        """Replace .run() with a recorder returning a valid `Answer` payload."""
        captured = {}

        def fake_run(messages, generation_kwargs=None, **kwargs):
            captured.update(generation_kwargs or {})
            return {"replies": [HaystackChatMessage.from_assistant('{"text": "ok"}')]}

        generator.run = fake_run
        return captured

    @override_settings(OPENAI_API_KEY="sk-test")
    async def test_responses_generator_gets_text_format(self):
        generator = openai_responses_chat(model="gpt-5-mini")
        captured = self._capture(generator)

        result = await Run(generator=generator).run([], response_format=Answer)

        assert captured == {"text_format": Answer}
        assert result == Answer(text="ok")

    @override_settings(OPENAI_API_KEY="sk-test")
    async def test_chat_generator_gets_response_format(self):
        generator = openai_chat(model="gpt-4o-mini")
        captured = self._capture(generator)

        await Run(generator=generator).run([], response_format=Answer)

        assert captured == {"response_format": Answer}

    def test_subclasses_inherit_through_the_mro(self):
        # Mistral and OpenRouter subclass OpenAIChatGenerator without being
        # installed here, so the lookup must walk the MRO rather than match a
        # class name exactly.
        class VendorChatGenerator(OpenAIChatGenerator):
            pass

        generator = VendorChatGenerator.__new__(VendorChatGenerator)

        assert schema_kwargs(generator, Answer) == {"response_format": Answer}

    def test_unlisted_generator_assumes_openai_compatible(self):
        assert schema_kwargs(object(), Answer) == {"response_format": Answer}

    @pytest.mark.parametrize(
        "class_name",
        ["OllamaChatGenerator", "AnthropicChatGenerator", "TransformersChatGenerator"],
    )
    def test_vendors_without_a_run_time_kwarg_raise(self, class_name):
        # Guessing `response_format` here would be a TypeError from the provider
        # client, or silently ignored - so say so instead.
        generator = type(class_name, (), {})()

        with pytest.raises(ValueError, match=f"{class_name} takes no structured-output kwarg"):
            schema_kwargs(generator, Answer)


class TestReasoningStream:
    """Reasoning models stream their summary separately from the answer."""

    @staticmethod
    async def _events(*chunks):
        """Feed StreamingChunks through Stream.get_events."""
        queue: asyncio.Queue = asyncio.Queue()
        for chunk in chunks:
            await queue.put(chunk)
        await queue.put(base._SENTINEL)
        stream = Stream.__new__(Stream)
        stream.citation_registry = None
        return [event async for event in stream.get_events(queue, None)]

    async def test_reasoning_chunk_becomes_an_event(self):
        # Haystack carries the summary on StreamingChunk.reasoning; without this
        # the protocol's reasoning parts never fire and the reasoning tokens are
        # paid for nothing.
        chunk = StreamingChunk(content="", index=0, reasoning=ReasoningContent(reasoning_text="because"))

        events = await self._events(chunk)

        assert [(e.event_type, e.content) for e in events] == [("reasoning_chunk", "because")]

    async def test_answer_and_reasoning_arrive_as_their_own_events(self):
        # A StreamingChunk may carry content or reasoning, never both, so the
        # summary and the answer reach the client as distinct events.
        events = await self._events(
            StreamingChunk(content="", index=0, reasoning=ReasoningContent(reasoning_text="because")),
            StreamingChunk(content="answer"),
        )

        assert [(e.event_type, e.content) for e in events] == [
            ("reasoning_chunk", "because"),
            ("text_chunk", "answer"),
        ]


class TestAgentHook:
    """Agent.llm / llm_kwargs / get_llm()."""

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_factory_attribute_is_not_bound_as_method(self):
        class FactoryAgent(Agent):
            name = "Factory Agent"
            model = "gpt-4o-mini"
            llm = openai_chat

        generator = FactoryAgent().get_llm()

        assert isinstance(generator, OpenAIChatGenerator)
        assert generator.model == "gpt-4o-mini"

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_defaults_to_responses_generator(self):
        class DefaultAgent(Agent):
            name = "Default Agent"
            model = "gpt-5-mini"

        assert isinstance(DefaultAgent().get_llm(), OpenAIResponsesChatGenerator)

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_llm_kwargs_become_generation_kwargs(self):
        class ReasoningAgent(Agent):
            name = "Reasoning Agent"
            model = "gpt-5-mini"
            llm = openai_responses_chat
            llm_kwargs = {"reasoning": {"effort": "low"}}

        agent = ReasoningAgent()

        assert agent.get_llm().generation_kwargs == {"reasoning": {"effort": "low"}}
        # Per-call kwargs win over the class attribute.
        overridden = agent.get_llm(generation_kwargs={"reasoning": {"effort": "high"}})
        assert overridden.generation_kwargs == {"reasoning": {"effort": "high"}}

    def test_generator_instance_is_rejected(self):
        class InstanceAgent(Agent):
            name = "Instance Agent"
            model = "gpt-4o-mini"
            llm = "not-a-factory"

        with pytest.raises(TypeError, match="must be a generator factory"):
            InstanceAgent().get_llm()
