---
title: Generators
type: docs
weight: 123
---

`django_ai_sdk.generators` holds one factory per Haystack chat generator class. Each
factory fills the provider credentials and base URL from vendor-specific Django
settings, then passes every other keyword straight through to the Haystack class.

```python
from django_ai_sdk.generators import openai_responses_chat

generator = openai_responses_chat(model="gpt-5-mini")
```

Factory names follow the Haystack class name minus `Generator`:
`OpenAIResponsesChatGenerator` → `openai_responses_chat`.

## Available factories

| Factory | Haystack class | Install |
| --- | --- | --- |
| `openai_chat` | `OpenAIChatGenerator` | included |
| `openai_responses_chat` | `OpenAIResponsesChatGenerator` | included |
| `azure_openai_chat` | `AzureOpenAIChatGenerator` | included |
| `azure_openai_responses_chat` | `AzureOpenAIResponsesChatGenerator` | included |
| `anthropic_chat` | `AnthropicChatGenerator` | `django-ai-sdk[anthropic]` |
| `mistral_chat` | `MistralChatGenerator` | `django-ai-sdk[mistral]` |
| `ollama_chat` | `OllamaChatGenerator` | `django-ai-sdk[ollama]` |
| `huggingface_api_chat` | `HuggingFaceAPIChatGenerator` | `django-ai-sdk[huggingface]` |
| `transformers_chat` | `TransformersChatGenerator` | `django-ai-sdk[huggingface]` |
| `openrouter_chat` | `OpenRouterChatGenerator` | `django-ai-sdk[openrouter]` |

A factory whose integration package is missing raises `ImportError` naming the extra
that provides it. See [Settings Reference](/manual/settings/#provider-credentials)
for the settings each vendor reads.

## Using a generator on an agent

Set `llm` to the factory itself — do not call it. `get_llm()` invokes it and supplies
the agent's model:

```python
from django_ai_sdk import Agent
from django_ai_sdk.generators import openai_responses_chat


class MyAgent(Agent):
    name = "My Agent"
    model = "gpt-5-mini"
    llm = openai_responses_chat

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm())
```

`llm` defaults to `openai_responses_chat`, so an agent that talks to OpenAI can omit
it. Switching provider is one line:

```python
class MyAgent(Agent):
    llm = anthropic_chat
```

## Reasoning and other generation parameters

The factories never enable reasoning on their own. Put the vendor's own parameters in
`llm_kwargs`; they are passed as Haystack's `generation_kwargs`:

```python
class MyAgent(Agent):
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "low", "summary": "auto"}}
```

```python
class MyAgent(Agent):
    llm = anthropic_chat
    llm_kwargs = {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096}
```

Ollama is the exception that proves the rule: `think` is an init parameter there, not a
generation kwarg, so `ollama_chat` lifts it out for you and `llm_kwargs` stays uniform:

```python
class MyAgent(Agent):
    llm = ollama_chat
    llm_kwargs = {"think": True}
```

Keyword arguments passed to `get_llm()` win over `llm_kwargs`, which lets a single
agent build a cheaper generator for a side task:

```python
generator = self.get_llm(generation_kwargs={})
```

## Structured output

`Agent.run(response_format=Schema)` and `Run.run(response_format=Schema)` are
vendor-neutral, but providers disagree on how a schema is passed. `Run` resolves that
through the `SCHEMA_KWARGS` table in `django_ai_sdk/generators/schema.py`, keyed by
generator class name - so resolving one vendor imports none of them - and walked over
the MRO, so subclasses inherit.

| Generator | How a schema is passed | Works with `response_format=` |
| --- | --- | --- |
| `OpenAIResponsesChatGenerator`, `AzureOpenAIResponsesChatGenerator` | `text_format` in `generation_kwargs`, a Pydantic model | yes |
| `OpenAIChatGenerator`, `AzureOpenAIChatGenerator` | `response_format` in `generation_kwargs`, a Pydantic model | yes |
| `MistralChatGenerator`, `OpenRouterChatGenerator` | inherited from `OpenAIChatGenerator` | yes |
| `OllamaChatGenerator` | `response_format` **at construction**, a JSON schema | no - see below |
| `AnthropicChatGenerator`, `HuggingFaceAPIChatGenerator`, `TransformersChatGenerator` | no schema parameter | no |

A generator in the last two rows raises `ValueError` rather than sending a keyword its
client would reject:

```
ValueError: OllamaChatGenerator takes no structured-output kwarg at run time.
```

Ollama wants the schema when the generator is built, as a JSON schema. Pass it as an
init kwarg through `get_llm()` and call `run()` without `response_format`:

```python
generator = self.get_llm(response_format=Answer.model_json_schema())
reply = await Run(generator=generator).run(messages)
answer = Answer.model_validate_json(reply)
```

For Anthropic and Hugging Face, ask for JSON in the prompt and validate the reply
yourself, or route that one call through an OpenAI-compatible generator.

An unlisted generator class is assumed OpenAI-compatible and gets `response_format`.

## Adding a vendor

Everything a vendor needs lives in its own module. To add DeepSeek:

1. Create `django_ai_sdk/generators/deepseek.py`. One module per vendor is what
   makes the extra optional, so import the integration at the top:

   ```python
   from haystack_integrations.components.generators.deepseek import DeepSeekChatGenerator

   from django_ai_sdk.generators.base import build_kwargs, resolve_secret, resolve_setting


   def deepseek_chat(**kwargs: Any) -> DeepSeekChatGenerator:
       """DeepSeek chat generator wired to Django settings."""
       return DeepSeekChatGenerator(
           **build_kwargs(
               {
                   "api_key": resolve_secret("DEEPSEEK_API_KEY"),
                   "api_base_url": resolve_setting("DEEPSEEK_API_URL"),
               },
               kwargs,
           )
       )
   ```

2. Add `deepseek = ["deepseek-haystack>=..."]` to `[project.optional-dependencies]`
   in `pyproject.toml`, and to the `all` extra.
3. Add `deepseek_chat` to `_LAZY` and `__all__` in
   `django_ai_sdk/generators/__init__.py` so it resolves on first use, and add a
   `SCHEMA_KWARGS` row in `schema.py` only if DeepSeek's structured-output keyword
   differs from `response_format`.

Nothing in `Run`, `Agent`, or the adapters changes.

## Token usage while streaming

The adapters never reconfigure a generator: what the factory built is what runs. That
matters for [token accounting](/manual/tracing/), because the two OpenAI APIs differ:

- `openai_responses_chat` reports usage on a streamed response without being asked.
- `openai_chat` (Chat Completions) reports none unless `stream_options` is requested,
  and that option is only valid on a streaming call - which is why it is not a factory
  default. Declare it on the agents that stream:

  ```python
  class MyAgent(Agent):
      llm = openai_chat
      llm_kwargs = {"stream_options": {"include_usage": True}}
  ```

Other vendors have no equivalent switch; a generator that reports no usage leaves the
token columns null.

## Custom factories

A factory is a plain callable returning a Haystack chat generator, so any function
works — including one that reads your own settings:

```python
from django.conf import settings
from django_ai_sdk.generators import openai_chat


def cheap_openai_chat(**kwargs):
    """Local fine-tune served behind an OpenAI-compatible endpoint."""
    kwargs.setdefault("api_base_url", settings.LOCAL_LLM_URL)
    return openai_chat(**kwargs)


class MyAgent(Agent):
    llm = cheap_openai_chat
```

A factory for a vendor that ships in its own package imports it at module scope,
like the built-in ones - keep the factory in its own module so importing it is what
requires the package:

```python
# myapp/generators/vertex.py
from haystack_integrations.components.generators.anthropic import (
    AnthropicVertexChatGenerator,
)

from django_ai_sdk.generators import build_kwargs


def vertex_chat(**kwargs):
    """Anthropic on Vertex AI - needs django-ai-sdk[anthropic]."""
    return AnthropicVertexChatGenerator(**build_kwargs({"region": "europe-west1"}, kwargs))
```

## Helpers

| Helper | Purpose |
| --- | --- |
| `merge_generation_kwargs(base, extra)` | Shallow merge where `extra` wins. |
| `schema_kwargs(generator, schema)` | The structured-output generation kwargs for that generator, from the `SCHEMA_KWARGS` table in `generators/schema.py`. `Run` calls it for you. |
