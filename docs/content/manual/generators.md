---
title: Generators
sidebar:
  order: 123
---

`django_ai_sdk.generators` holds one factory per Haystack chat generator class. Each
factory fills the provider credentials and base URL from vendor-specific Django
settings, then passes every other keyword straight through to the Haystack class.

```python
from django_ai_sdk.generators import openai_responses_chat

generator = openai_responses_chat(model="gpt-5-mini")
```

Each factory name is the Haystack class name without `Generator`, in snake case. For example,
`OpenAIResponsesChatGenerator` becomes `openai_responses_chat`.

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
that provides it. See [Settings Reference](settings.md#provider-credentials)
for the settings each vendor reads.

## Using a generator on an agent

Set `llm` to the factory itself and do not call it. `get_llm()` calls it and passes
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
it. Switching provider takes one line:

```python
class MyAgent(Agent):
    llm = anthropic_chat
```

## Reasoning and other generation parameters

The factories never enable reasoning on their own. Put the vendor's own parameters in
`llm_kwargs`. The SDK passes them as Haystack's `generation_kwargs`:

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

Ollama is the exception. There `think` is an init parameter, not a generation kwarg.
`ollama_chat` moves it out of `llm_kwargs`, so `llm_kwargs` works the same for every vendor:

```python
class MyAgent(Agent):
    llm = ollama_chat
    llm_kwargs = {"think": True}
```

Keyword arguments passed to `get_llm()` override `llm_kwargs`. One agent can use this to
build a cheaper generator for a side task:

```python
generator = self.get_llm(generation_kwargs={})
```

## Structured output

`Agent.run(response_format=Schema)` and `Run.run(response_format=Schema)` are
vendor-neutral, but each provider takes the schema in a different way. `Run` looks the
keyword up in the `SCHEMA_KWARGS` table in `django_ai_sdk/generators/schema.py`. The
table is keyed by generator class name, so resolving one vendor imports none of them.
`Run` walks it over the MRO, so subclasses inherit their parent's row.

| Generator | How a schema is passed | Works with `response_format=` |
| --- | --- | --- |
| `OpenAIResponsesChatGenerator`, `AzureOpenAIResponsesChatGenerator` | `text_format` in `generation_kwargs`, a Pydantic model | yes |
| `OpenAIChatGenerator`, `AzureOpenAIChatGenerator` | `response_format` in `generation_kwargs`, a Pydantic model | yes |
| `MistralChatGenerator`, `OpenRouterChatGenerator` | inherited from `OpenAIChatGenerator` | yes |
| `OllamaChatGenerator` | `response_format` **at construction**, a JSON schema | no, see below |
| `AnthropicChatGenerator`, `HuggingFaceAPIChatGenerator`, `TransformersChatGenerator` | no schema parameter | no |

A generator in the last two rows raises `ValueError` instead of sending a keyword its
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
yourself. You can also route that one call through an OpenAI-compatible generator.

`Run` treats an unlisted generator class as OpenAI-compatible and passes it `response_format`.

### Structured output from a tool loop

The SDK does not support `agent.run(tools=True, response_format=Schema)`, by design.
`response_format` is a generation kwarg, as the table above shows. A Haystack `Agent`'s
tool loop applies its generation kwargs to *every* step, including the steps before the
final answer. Structured output for the whole run would force an early step, such as
"should I call `search_web`?", into your final answer's schema. The generators in the
last two rows of the table accept no schema kwarg at any step. Passing both does not
raise. `response_format` wins and the run skips tools entirely, see
[Tools](../serving/protocols.md#tools).

The supported approach uses two calls. Run the tool loop free-form, then make one more
call without tools to put the finished text into the schema:

```python
text = await agent.run(messages, tools=True)
report = await Run(generator=agent.get_llm()).run(
    [ChatMessage(role="assistant", content=text)],
    response_format=ResearchSummary,
)
```

## Adding a vendor

Everything a vendor needs is in its own module. To add DeepSeek:

:::steps

1. Create `django_ai_sdk/generators/deepseek.py`. One module per vendor keeps the
   extra optional, so import the integration at the top:

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
:::

Nothing in `Run`, `Agent`, or the adapters changes.

## Token usage while streaming

The adapters never reconfigure a generator. The generator the factory built is the one
that runs. This matters for [token accounting](tracing.md), because the two OpenAI APIs differ:

- `openai_responses_chat` reports usage on a streamed response by default.
- `openai_chat` (Chat Completions) reports no usage unless you request `stream_options`.
  That option is only valid on a streaming call, so it is not a factory default.
  Declare it on the agents that stream:

  ```python
  class MyAgent(Agent):
      llm = openai_chat
      llm_kwargs = {"stream_options": {"include_usage": True}}
  ```

Other vendors have no equivalent switch. A generator that reports no usage leaves the
token columns null.

## Custom factories

A factory is a plain callable that returns a Haystack chat generator. Any function
works, including one that reads your own settings:

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

A factory for a vendor that ships in its own package imports that package at module
scope, like the built-in factories. Keep the factory in its own module, so only
importing that module requires the package:

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
