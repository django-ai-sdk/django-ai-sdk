from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import openai
from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist, SuspiciousOperation
from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.core.exceptions import ValidationError as DjangoValidationError
from django.http import Http404
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _
from pydantic import ValidationError

from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Haystack wraps provider errors in PipelineRuntimeError one or two frames deep.
_MAX_CAUSE_DEPTH = 5


class ErrorCode(StrEnum):
    CONTEXT_OVERFLOW = "context_overflow"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    CONFIGURATION_ERROR = "configuration_error"
    CONTENT_FILTERED = "content_filtered"
    KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"
    TOOL_FAILED = "tool_failed"
    FILE_UNSUPPORTED = "file_unsupported"
    FILE_PROCESSING_FAILED = "file_processing_failed"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INVALID_REQUEST = "invalid_request"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    message: Any
    status: int = 500
    retryable: bool = False


_REGISTRY: dict[str, ErrorSpec] = {}


def register_error_code(
    code: str, *, message: Any, status: int = 500, retryable: bool = False
) -> None:
    """Add a code, or change an existing one's message, status or retryable flag.
    Call it from `AppConfig.ready()`."""
    _REGISTRY[str(code)] = ErrorSpec(str(code), message, status, retryable)


def get_error_spec(code: str | None) -> ErrorSpec:
    return _REGISTRY.get(str(code or ""), _REGISTRY[ErrorCode.UNKNOWN])


register_error_code(
    ErrorCode.CONTEXT_OVERFLOW,
    message=_("This conversation is too big for the model. Start a new chat."),
    status=400,
)
register_error_code(
    ErrorCode.RATE_LIMITED,
    message=_("Too many requests right now. Please try again in a moment."),
    status=429,
    retryable=True,
)
register_error_code(
    ErrorCode.PROVIDER_UNAVAILABLE,
    message=_("The AI service is temporarily unavailable. Please try again in a moment."),
    status=503,
    retryable=True,
)
register_error_code(
    ErrorCode.PROVIDER_TIMEOUT,
    message=_("The AI service took too long to answer. Try again, or ask for less at once."),
    status=504,
    retryable=True,
)
register_error_code(
    ErrorCode.CONFIGURATION_ERROR,
    message=_("This assistant is not configured correctly. Please contact an administrator."),
)
register_error_code(
    ErrorCode.CONTENT_FILTERED,
    message=_(
        "The request was blocked by the provider's content filter. Try rephrasing your message."
    ),
    status=400,
)
register_error_code(
    ErrorCode.KNOWLEDGE_UNAVAILABLE,
    message=_("The knowledge base is temporarily unavailable. Please try again in a moment."),
    status=503,
    retryable=True,
)
register_error_code(
    ErrorCode.TOOL_FAILED,
    message=_("A tool the assistant used failed. The answer may be incomplete."),
    retryable=True,
)
register_error_code(
    ErrorCode.FILE_UNSUPPORTED,
    message=_("This file type can't be read, or the file is empty."),
    status=400,
)
register_error_code(
    ErrorCode.FILE_PROCESSING_FAILED,
    message=_("This file could not be processed. Try uploading it again."),
    retryable=True,
)
register_error_code(
    ErrorCode.PERMISSION_DENIED, message=_("You don't have permission to do this."), status=403
)
register_error_code(ErrorCode.NOT_FOUND, message=_("This item could not be found."), status=404)
register_error_code(
    ErrorCode.CONFLICT, message=_("This conflicts with something that already exists."), status=409
)
register_error_code(ErrorCode.INVALID_REQUEST, message=_("The request was invalid."), status=400)
register_error_code(
    ErrorCode.BLOCKED, message=_("This assistant's policy stopped this answer."), status=400
)
register_error_code(
    ErrorCode.UNKNOWN,
    message=_("Something went wrong. Please try again."),
    retryable=True,
)


class AiSdkError(Exception):
    """`detail` is for the log only; clients get the code's message."""

    code: str = ErrorCode.UNKNOWN

    def __init__(self, detail: str = "", code: str | None = None) -> None:
        super().__init__(detail)
        if code is not None:
            self.code = code


class NotFound(AiSdkError, ValueError):
    """A looked-up object doesn't exist (or isn't visible to the user)."""

    code = ErrorCode.NOT_FOUND


class UserError(AiSdkError, ValueError):
    """Unlike any other exception, its message is sent to the client."""

    code = ErrorCode.INVALID_REQUEST


class Blocked(AiSdkError):
    """Raised by `Agent.run()` when the policy blocks the question or withholds the answer.
    `stage` is "gate" or "finish"."""

    code = ErrorCode.BLOCKED

    def __init__(self, message: str, *, stage: str = "", turn_id: str = "") -> None:
        super().__init__(f"blocked by oversight at {stage or 'unknown stage'}")
        self.message = message
        self.stage = stage
        self.turn_id = turn_id


# Providers report context overflow as a 400 with prose, and OpenAI-compatible
# servers (vLLM, llama.cpp, ...) often without `code=context_length_exceeded`.
_CONTEXT_OVERFLOW_TEXT = (
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "reduce the length of the messages",
    "exceeds the context window",
)


def _is_context_overflow(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(needle in text for needle in _CONTEXT_OVERFLOW_TEXT)


def _classify_openai(exc: BaseException) -> str | None:
    if isinstance(exc, openai.ContentFilterFinishReasonError):
        return ErrorCode.CONTENT_FILTERED
    if isinstance(exc, openai.RateLimitError):
        if exc.code == "insufficient_quota":
            return ErrorCode.CONFIGURATION_ERROR
        return ErrorCode.RATE_LIMITED
    # Before APIConnectionError, its base class.
    if isinstance(exc, openai.APITimeoutError):
        return ErrorCode.PROVIDER_TIMEOUT
    if isinstance(exc, openai.APIConnectionError):
        return ErrorCode.PROVIDER_UNAVAILABLE
    # A 404 without an error body comes from a gateway (LiteLLM, vLLM, a hosted router)
    # with no model behind it right now; a wrong model name comes with one.
    if isinstance(exc, openai.NotFoundError) and not isinstance(exc.body, dict):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if isinstance(
        exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)
    ):
        return ErrorCode.CONFIGURATION_ERROR
    if isinstance(exc, openai.BadRequestError):
        if exc.code == "context_length_exceeded" or _is_context_overflow(exc):
            return ErrorCode.CONTEXT_OVERFLOW
        if exc.code in ("content_filter", "content_policy_violation"):
            return ErrorCode.CONTENT_FILTERED
        if (exc.param or "").startswith("tools"):
            return ErrorCode.CONFIGURATION_ERROR
        return None
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return ErrorCode.PROVIDER_UNAVAILABLE
    return None


def _classify_anthropic(exc: BaseException) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    if isinstance(exc, anthropic.RateLimitError):
        return ErrorCode.RATE_LIMITED
    if isinstance(exc, anthropic.APITimeoutError):
        return ErrorCode.PROVIDER_TIMEOUT
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if isinstance(exc, anthropic.NotFoundError) and not isinstance(exc.body, dict):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if isinstance(
        exc,
        (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError),
    ):
        return ErrorCode.CONFIGURATION_ERROR
    if isinstance(exc, anthropic.BadRequestError):
        return ErrorCode.CONTEXT_OVERFLOW if _is_context_overflow(exc) else None
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return ErrorCode.PROVIDER_UNAVAILABLE
    return None


_PROVIDER_CLASSIFIERS = (_classify_openai, _classify_anthropic)


def _classify_with_hooks(exc: BaseException) -> str | None:
    classifiers: list[Callable[[BaseException], str | None]] = [
        import_string(path) for path in resolve_setting("AI_SDK_ERROR_CLASSIFIERS", [])
    ]
    for classifier in classifiers:
        try:
            code = classifier(exc)
        except Exception:
            logger.opt(exception=True).error("Error classifier {} failed", classifier)
            continue
        if code is not None:
            return code
    return None


def classify_error(exc: BaseException, _depth: int = 0) -> str:
    if isinstance(exc, AiSdkError):
        return exc.code
    if (code := _classify_with_hooks(exc)) is not None:
        return code
    if isinstance(exc, (PermissionDenied, DjangoPermissionDenied)):
        return ErrorCode.PERMISSION_DENIED
    if isinstance(exc, (Http404, ObjectDoesNotExist)):
        return ErrorCode.NOT_FOUND
    if isinstance(exc, ConflictError):
        return ErrorCode.CONFLICT
    if isinstance(exc, ImproperlyConfigured):
        return ErrorCode.CONFIGURATION_ERROR

    for classify in _PROVIDER_CLASSIFIERS:
        if (code := classify(exc)) is not None:
            return code
    if isinstance(exc, TimeoutError):
        return ErrorCode.PROVIDER_TIMEOUT
    if isinstance(exc, ConnectionError):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if _is_context_overflow(exc):
        return ErrorCode.CONTEXT_OVERFLOW

    cause = exc.__cause__ or exc.__context__
    if cause is not None and _depth < _MAX_CAUSE_DEPTH:
        return classify_error(cause, _depth + 1)
    return ErrorCode.UNKNOWN


def new_ref() -> str:
    return secrets.token_hex(4)


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    ref: str

    @property
    def spec(self) -> ErrorSpec:
        return get_error_spec(self.code)

    def as_dict(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "code": spec.code,
            "message": str(spec.message),
            "retryable": spec.retryable,
            "ref": self.ref,
        }


def describe_error(exc: BaseException) -> ErrorInfo:
    return ErrorInfo(code=classify_error(exc), ref=new_ref())


def public_detail(exc: BaseException) -> str:
    if isinstance(exc, AiSdkError):
        return str(exc)
    info = describe_error(exc)
    logger.warning("Recorded as error ref=%s: %s: %s", info.ref, type(exc).__name__, exc)
    return f"{info.spec.message} (ref {info.ref})"


_BAD_INPUT = (ValidationError, DjangoValidationError, SuspiciousOperation)


def error_response(exc: BaseException) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, _BAD_INPUT):
        info = ErrorInfo(code=ErrorCode.INVALID_REQUEST, ref=new_ref())
    else:
        info = describe_error(exc)

    status = info.spec.status
    if status >= 500:
        logger.opt(exception=exc).error("Request failed [{}] {}: {}", info.ref, info.code, exc)
    else:
        logger.info("Request rejected [{}] {}: {}", info.ref, info.code, exc)

    body = info.as_dict()
    if isinstance(exc, UserError) and str(exc):
        body["message"] = str(exc)
    if isinstance(exc, Blocked) and exc.message:
        body["message"] = exc.message
    if isinstance(exc, ValidationError):
        body["errors"] = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
    return status, body
