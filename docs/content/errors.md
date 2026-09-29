---
title: Error Handling
type: docs
weight: 6
---

When something fails, a client gets a stable **error code**, a message and a short **ref**. It never gets the exception text: provider URLs, file paths and class names stay in your logs, tagged with the same ref, so a user can quote it to an admin.

## Error codes

Branch on the code, not on the message. `retryable` tells the client whether trying again can help; the message says what the user can do.

| Code | Status | Retryable | Meaning |
| --- | --- | --- | --- |
| `context_overflow` | 400 | no | The conversation no longer fits the model's context window |
| `rate_limited` | 429 | yes | The provider is rate limiting |
| `provider_unavailable` | 503 | yes | The provider is down or unreachable, or a gateway has no model behind it right now (a 404 without an error body) |
| `provider_timeout` | 504 | yes | The provider took longer than the generator's timeout, often on a very large prompt |
| `configuration_error` | 500 | no | Bad API key, unknown model, invalid tool schema, exhausted quota, missing setting |
| `content_filtered` | 400 | no | The provider's content filter blocked the request |
| `knowledge_unavailable` | 503 | yes | A vector store (RAG) failed or is locked |
| `tool_failed` | 500 | yes | A tool call failed (on tool parts only) |
| `file_unsupported` | 400 | no | An uploaded file is empty or of a type no pipeline reads |
| `file_processing_failed` | 500 | yes | An uploaded file could not be processed |
| `permission_denied` | 403 | no | The user may not do this |
| `not_found` | 404 | no | The object doesn't exist, or isn't visible to the user |
| `conflict` | 409 | no | The request conflicts with existing state |
| `invalid_request` | 400 | no | The input was invalid |
| `unknown` | 500 | yes | Anything else |

`django_ai_sdk.errors.ErrorCode` holds these. The default messages are translatable (`gettext_lazy`).

## Raising a code

Provider errors (`openai`, and `anthropic` when installed) are classified by type. Raise `AiSdkError` when your own code knows what went wrong; its message goes to the log only:

```python
from django_ai_sdk.errors import AiSdkError, ErrorCode, NotFound

raise AiSdkError(f"Search index {path} is locked", ErrorCode.KNOWLEDGE_UNAVAILABLE)
raise NotFound("Thread not found")  # also a ValueError
```

A message written for the user, like a validation failure, is the one exception to the rule. Raise `UserError` and its message is sent as the response's `message`:

```python
from django_ai_sdk.errors import UserError

raise UserError("File too large. Maximum size is 10 MB.")
```

Any other exception, a bare `ValueError` included, is a 500 with code `unknown`, logged with its traceback.

## Your own codes

Register a code from your `AppConfig.ready()`, then raise it:

```python
from django.utils.translation import gettext_lazy as _
from django_ai_sdk.errors import AiSdkError, register_error_code

register_error_code(
    "quota_exceeded",
    message=_("Your plan's monthly quota is used up."),
    status=402,
    retryable=False,
)

raise AiSdkError(f"Org {org.id} is over quota", "quota_exceeded")
```

`register_error_code` also changes a built-in code's message, status or retryable flag. A code nobody registered is sent as `unknown`.

## Other providers

For errors the SDK doesn't recognise, list classifiers in `AI_SDK_ERROR_CLASSIFIERS`. Each takes the exception and returns a code, or `None` to pass. They run before the SDK's own rules, on the exception and on its causes:

```python
# myapp/errors.py
from botocore.exceptions import ClientError
from django_ai_sdk.errors import ErrorCode


def classify_bedrock(exc):
    if isinstance(exc, ClientError):
        if exc.response["Error"]["Code"] == "ThrottlingException":
            return ErrorCode.RATE_LIMITED
        return ErrorCode.PROVIDER_UNAVAILABLE
    return None
```

```python
AI_SDK_ERROR_CLASSIFIERS = ["myapp.errors.classify_bedrock"]
```

## In a chat stream

**Vercel protocol:** a failed turn sends a `data-error` part and then the standard `error` part:

```json
{"type": "data-error", "data": {"code": "rate_limited", "message": "Too many requests right now. Please try again in a moment.", "retryable": true, "ref": "3f9a1c02"}}
{"type": "error", "errorText": "rate_limited"}
```

Show `message`, or your own copy keyed on `code`. The `data-error` part is stored with the reply, so a reloaded thread shows the same error. A failed tool call becomes a `tool-output-error` part with `errorText: "tool_failed"`.

**OpenAI protocol:** `{"error": {"message": "<message>", "type": "server_error", "code": "rate_limited", "retryable": true, "ref": "3f9a1c02"}}`.

## In your API views

`error_response(exc)` turns any exception into a status and a JSON body. Register it once instead of catching exceptions in every view:

```python
# django-ninja
from django_ai_sdk.contrib.ninja import register_error_handlers

api = NinjaAPI()
register_error_handlers(api)
```

```python
# Django REST framework
REST_FRAMEWORK = {"EXCEPTION_HANDLER": "django_ai_sdk.contrib.drf.exception_handler"}
```

The body matches `django_ai_sdk.views.schemas.ErrorResponse`:

```json
{"code": "not_found", "message": "This item could not be found.", "retryable": false, "ref": "3f9a1c02"}
```

A pydantic `ValidationError` also returns its field `errors`. Ninja's and DRF's own exceptions keep their own handling.

## What the model sees

- A tool that raises gives the model the error text, as Haystack does, so it can fix its arguments or pick another tool. The model may repeat that text to the user; if your tools' errors can carry secrets, return a safe error instead of raising, or rewrite results in an `after_tool` hook.
- A tool result longer than `AI_SDK_TOOL_OUTPUT_LIMIT` (100,000 characters, about 25k tokens) is cut to its start, with a note up front saying how much the model sees. Set `ToolAgentConfig.max_tool_output_chars` per agent, or `0` to turn it off. The limit is per result: a run that gathers many results can still overflow the context window, which ends the turn with `context_overflow`.

## Vector stores

A `persistent` (local path) Qdrant store is locked by the process that opens it. `QdrantStorageConfig.lock_timeout` (default 10 seconds) is how long a chat turn or a document task waits for another process to release it; after that it fails with `knowledge_unavailable`. Run a Qdrant server (`AI_SDK_VECTOR_STORE_URL`) whenever several processes serve requests or run tasks.
