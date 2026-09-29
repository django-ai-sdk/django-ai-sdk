from __future__ import annotations

import json

import anthropic
import httpx
import openai
import pytest
from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from django_ai_sdk import errors
from django_ai_sdk.errors import (
    AiSdkError,
    ErrorCode,
    NotFound,
    UserError,
    classify_error,
    error_response,
    get_error_spec,
    register_error_code,
)
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from django_ai_sdk.protocols.vercel import error_data

SECRET = "/var/www/SECRET/stores/qdrant"
REQUEST = httpx.Request("POST", "https://llm.example/v1/chat/completions")


def status_error(cls, status: int, body: dict | None = None):
    return cls(f"{SECRET} upstream said no", response=httpx.Response(status, request=REQUEST), body=body)


class TestClassifyError:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (status_error(openai.RateLimitError, 429), ErrorCode.RATE_LIMITED),
            (
                status_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}),
                ErrorCode.CONFIGURATION_ERROR,
            ),
            (openai.APIConnectionError(request=REQUEST), ErrorCode.PROVIDER_UNAVAILABLE),
            (openai.APITimeoutError(request=REQUEST), ErrorCode.PROVIDER_TIMEOUT),
            (status_error(openai.InternalServerError, 503), ErrorCode.PROVIDER_UNAVAILABLE),
            (status_error(openai.AuthenticationError, 401), ErrorCode.CONFIGURATION_ERROR),
            (
                status_error(openai.NotFoundError, 404, {"code": "model_not_found"}),
                ErrorCode.CONFIGURATION_ERROR,
            ),
            (status_error(openai.NotFoundError, 404), ErrorCode.PROVIDER_UNAVAILABLE),
            (status_error(anthropic.NotFoundError, 404), ErrorCode.PROVIDER_UNAVAILABLE),
            (
                status_error(openai.BadRequestError, 400, {"code": "context_length_exceeded"}),
                ErrorCode.CONTEXT_OVERFLOW,
            ),
            (
                status_error(openai.BadRequestError, 400, {"param": "tools[3].function.name"}),
                ErrorCode.CONFIGURATION_ERROR,
            ),
            (status_error(openai.BadRequestError, 400), ErrorCode.UNKNOWN),
            (TimeoutError(), ErrorCode.PROVIDER_TIMEOUT),
            (ConnectionResetError(), ErrorCode.PROVIDER_UNAVAILABLE),
            (PermissionDenied(), ErrorCode.PERMISSION_DENIED),
            (NotFound("Thread not found"), ErrorCode.NOT_FOUND),
            (AiSdkError("store locked", ErrorCode.KNOWLEDGE_UNAVAILABLE), ErrorCode.KNOWLEDGE_UNAVAILABLE),
            (ImproperlyConfigured("AI_SDK_X missing"), ErrorCode.CONFIGURATION_ERROR),
        ],
    )
    def test_typed(self, exc, expected):
        assert classify_error(exc) == expected

    def test_context_overflow_without_a_code(self):
        """OpenAI-compatible servers (vLLM) often send no `code`, only prose."""
        exc = openai.BadRequestError(
            "This model's maximum context length is 131072 tokens",
            response=httpx.Response(400, request=REQUEST),
            body=None,
        )
        assert classify_error(exc) == ErrorCode.CONTEXT_OVERFLOW

    @pytest.mark.parametrize(
        "message", ["Qdrant returned an error", "Collection doesn't exist", "too many tokens per min"]
    )
    def test_no_guessing_from_prose(self, message):
        assert classify_error(RuntimeError(message)) == ErrorCode.UNKNOWN

    def test_unwraps_a_wrapped_cause(self):
        try:
            try:
                raise openai.APIConnectionError(request=REQUEST)
            except openai.APIConnectionError as cause:
                raise RuntimeError("Pipeline failed") from cause
        except RuntimeError as wrapped:
            assert classify_error(wrapped) == ErrorCode.PROVIDER_UNAVAILABLE

    def test_survives_a_reference_cycle(self):
        first, second = Exception("one"), Exception("two")
        first.__cause__, second.__cause__ = second, first
        assert classify_error(first) == ErrorCode.UNKNOWN

    def test_not_found_is_still_a_value_error(self):
        assert isinstance(NotFound("x"), ValueError)


class TestErrorResponse:
    @pytest.mark.parametrize(
        ("exc", "status", "code"),
        [
            (NotFound(f"Thread {SECRET} not found"), 404, "not_found"),
            (PermissionDenied(f"No access to {SECRET}"), 403, "permission_denied"),
            (ConflictError(f"{SECRET} exists"), 409, "conflict"),
            (ValueError(f"bad {SECRET}"), 500, "unknown"),
            (json.JSONDecodeError(SECRET, "{", 0), 500, "unknown"),
            (RuntimeError(f"crash in {SECRET}"), 500, "unknown"),
            (status_error(openai.RateLimitError, 429), 429, "rate_limited"),
        ],
    )
    def test_body_is_a_code_never_the_text(self, exc, status, code):
        got_status, body = error_response(exc)

        assert got_status == status
        assert body["code"] == code
        assert body["message"] and body["ref"]
        assert "SECRET" not in json.dumps(body)

    def test_a_user_error_keeps_its_message(self):
        status, body = error_response(UserError("File too large. Maximum size is 10 MB."))

        assert status == 400
        assert body["code"] == "invalid_request"
        assert body["message"] == "File too large. Maximum size is 10 MB."

    def test_an_oversized_request_body_is_bad_input(self):
        from django.core.exceptions import RequestDataTooBig

        assert error_response(RequestDataTooBig("too big"))[0] == 400

    def test_an_unexpected_error_is_logged_with_its_traceback(self):
        """A ValueError from inside the SDK is a bug, not the caller's bad input."""
        records = []
        sink = errors.logger.add(records.append, level="ERROR")
        try:
            error_response(json.JSONDecodeError("Expecting value", "", 0))
        finally:
            errors.logger.remove(sink)

        [record] = records
        assert record.record["exception"] is not None

    def test_invalid_input_returns_its_field_errors(self):
        class Payload(BaseModel):
            count: int

        with pytest.raises(PydanticValidationError) as caught:
            Payload.model_validate({"count": "x"})
        status, body = error_response(caught.value)

        assert status == 400
        assert body["code"] == "invalid_request"
        assert body["errors"][0]["loc"] == ["count"]


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(errors, "_REGISTRY", dict(errors._REGISTRY))


class TestRegistry:
    def test_an_app_adds_its_own_code(self, registry):
        register_error_code("quota_exceeded", message="Your plan's quota is used up.", status=402)

        status, body = error_response(AiSdkError("org 7 over quota", "quota_exceeded"))

        assert status == 402
        assert body == {
            "code": "quota_exceeded",
            "message": "Your plan's quota is used up.",
            "retryable": False,
            "ref": body["ref"],
        }

    def test_an_app_code_streams_its_message(self, registry):
        register_error_code("quota_exceeded", message="Your plan's quota is used up.", status=402)

        assert error_data("quota_exceeded", "r1")["message"] == "Your plan's quota is used up."

    def test_an_app_changes_a_built_in_message(self, registry):
        register_error_code(ErrorCode.NOT_FOUND, message="Gone.", status=404)

        assert error_response(NotFound("x"))[1]["message"] == "Gone."

    def test_an_unregistered_code_is_unknown(self):
        spec = get_error_spec("never_registered")

        assert spec.code == "unknown"
        assert error_response(AiSdkError("x", "never_registered"))[1]["code"] == "unknown"


def classify_bedrock(exc):
    return ErrorCode.RATE_LIMITED if "ThrottlingException" in str(exc) else None


def broken_classifier(exc):
    raise RuntimeError("classifier bug")


class TestClassifierHooks:
    def test_a_hook_classifies_what_the_sdk_doesnt_know(self, settings):
        settings.AI_SDK_ERROR_CLASSIFIERS = ["tests.unit.test_errors.classify_bedrock"]

        assert classify_error(RuntimeError("ThrottlingException: slow down")) == ErrorCode.RATE_LIMITED
        assert classify_error(RuntimeError("other")) == ErrorCode.UNKNOWN

    def test_a_broken_hook_is_skipped(self, settings):
        settings.AI_SDK_ERROR_CLASSIFIERS = [
            "tests.unit.test_errors.broken_classifier",
            "tests.unit.test_errors.classify_bedrock",
        ]

        assert classify_error(RuntimeError("ThrottlingException")) == ErrorCode.RATE_LIMITED
