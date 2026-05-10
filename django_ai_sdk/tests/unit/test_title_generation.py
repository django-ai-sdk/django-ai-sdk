"""
Unit tests for thread title generation on Assistant.

Covers the override surface (``_build_title_prompt``, ``_fallback_thread_title``)
and the orchestration in ``generate_thread_title`` with the openai client
mocked. Does not exercise the real network — the SDK's title generation is
hardcoded to openai-compatible endpoints, see the TODO in assistant.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk import Assistant
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter


def _make_assistant(**overrides) -> Assistant:
    """Concrete Assistant subclass for tests — only what's needed to instantiate."""

    class _TestAssistant(Assistant):
        name = "test"
        model = "test-model"
        instructions = ["Be helpful"]
        protocol = VercelProtocolHandler
        storage_adapter = MemoryStorageAdapter

        async def get_pipeline_adapter(self, thread_id=None, storage_adapter=None):
            raise NotImplementedError

    for key, value in overrides.items():
        setattr(_TestAssistant, key, value)
    return _TestAssistant()


def _mock_openai_response(content: str | None) -> MagicMock:
    """Build a fake openai chat-completion response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _fallback_thread_title
# ---------------------------------------------------------------------------


def test_fallback_uses_first_user_message():
    a = _make_assistant()
    context = [("user", "Hoe laat is het?"), ("assistant", "12:00")]
    assert a._fallback_thread_title(context) == "Hoe laat is het?"


def test_fallback_truncates_long_user_message():
    a = _make_assistant()
    long_msg = "a" * 80
    title = a._fallback_thread_title([("user", long_msg)])
    assert title == "a" * 50 + "…"


def test_fallback_returns_default_when_no_user_content():
    a = _make_assistant(title_fallback_default="Nieuw gesprek")
    assert a._fallback_thread_title([("assistant", "hi")]) == "Nieuw gesprek"
    assert a._fallback_thread_title([("user", "")]) == "Nieuw gesprek"
    assert a._fallback_thread_title([]) == "Nieuw gesprek"


# ---------------------------------------------------------------------------
# title_prompt_template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_title_prompt_template_is_used():
    """Subclass override of title_prompt_template flows through to the LLM call."""
    a = _make_assistant(
        title_prompt_template="My custom prompt with chat: {chat_history}"
    )
    fake_response = _mock_openai_response("title")
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        await a.generate_thread_title([("user", "hello")])

    sent_prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert sent_prompt == "My custom prompt with chat: user: hello"


# ---------------------------------------------------------------------------
# generate_thread_title (mocked openai client)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_parsed_title_from_llm():
    a = _make_assistant()
    fake_response = _mock_openai_response('  "📉 Stock Trends"  ')
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        title = await a.generate_thread_title([("user", "Tell me about stocks")])

    assert title == "📉 Stock Trends"  # quotes/whitespace stripped


@pytest.mark.asyncio
async def test_generate_preserves_trailing_punctuation():
    """Regression: title generation must not strip trailing punctuation."""
    a = _make_assistant()
    fake_response = _mock_openai_response("What is I.T.?")
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        title = await a.generate_thread_title([("user", "I.T. question")])

    assert title == "What is I.T.?"


@pytest.mark.asyncio
async def test_generate_returns_none_when_llm_returns_empty_content():
    a = _make_assistant()
    fake_response = _mock_openai_response(None)  # reasoning model exhausted budget
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        title = await a.generate_thread_title([("user", "hello")])

    assert title is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_llm_raises():
    a = _make_assistant()
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        title = await a.generate_thread_title([("user", "hello")])

    assert title is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_no_context():
    a = _make_assistant()
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock()

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        assert await a.generate_thread_title([]) is None
        assert await a.generate_thread_title([("user", "")]) is None

    fake_client.chat.completions.create.assert_not_called()
