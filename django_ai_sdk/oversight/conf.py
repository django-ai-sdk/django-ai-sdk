"""The AI_SDK_OVERSIGHT setting. Only this module reads it; docs/content/manual/oversight.md lists the keys."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from django.apps import apps
from django.conf import settings
from django.core.checks import Error
from django.core.exceptions import ImproperlyConfigured
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from django_ai_sdk.oversight.types import Capture

APP_LABEL = "django_ai_sdk_oversight"

# Allowed values, shared with policy.py, which checks a policy's own options.
STREAM_MODES = ("auto", "live")
FAIL_MODES = ("closed", "open")

Positive = Annotated[float, Field(gt=0, strict=True)]
Retention = dict[Literal["content", "metadata"], Annotated[StrictInt, Field(ge=1)]]


class OversightSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Kill switch: False leaves every agent unguarded.
    ENABLED: StrictBool = True
    # Policy for agents without `oversight`; "" leaves them unguarded.
    DEFAULT_POLICY: StrictStr = ""
    # The most content the record keeps. A policy may keep less, never more.
    CAPTURE: Capture = Capture.MASKED
    STREAM_MODE: Literal["auto", "live"] = "auto"
    # Judges by alias: {"judge": {"class": "LLMJudge"}}.
    JUDGES: dict[str, Any] = {}
    # Milliseconds per inline stage; a lens still running at the deadline fails.
    DEADLINE_MS: dict[Literal["gate", "tool", "finish"], Positive | None] = {
        "gate": 1000,
        "tool": 1000,
        "finish": 2_000,
    }
    # Longer questions are refused at the gate: lenses would only read part of them.
    MAX_INPUT_CHARS: Positive | None = 100_000
    CLIENT_DETAIL: Literal["actions", "full"] = "actions"
    FAIL_MODE: Literal["closed", "open"] = "closed"
    RETENTION: Retention = {"content": 90, "metadata": 730}
    # Dotted paths or Notifier instances.
    NOTIFIERS: list[Any] = []
    # How long each process caches authored policies. 0 reads them every turn.
    RUNTIME_CACHE_SECONDS: Annotated[float, Field(ge=0, strict=True)] = 10

    @field_validator("JUDGES")
    @classmethod
    def _judge_specs(cls, value: dict[str, Any]) -> dict[str, Any]:
        from django_ai_sdk.oversight.judges.base import Judge

        bad = [alias for alias, spec in value.items() if not isinstance(spec, str | dict | Judge)]
        if bad:
            raise ValueError(f"aliases {bad} are not judge specs")
        return value


class AppSettings:
    """One settings dict, with its keys and defaults declared as a pydantic model.

    Call it with a key for `settings.<name>[key]`, or its default; a dict value merges
    one level deep into a dict default. The app's E012 check reports a bad dict.
    """

    def __init__(self, name: str, model: type[BaseModel]) -> None:
        self.name = name
        self.model = model
        self.defaults = {key: field.default for key, field in model.model_fields.items()}

    def _configured(self) -> Any:
        return getattr(settings, self.name, None) or {}

    @property
    def broken(self) -> bool:
        """Whether the setting is not a dict at all. Checks that read it skip."""
        return not isinstance(self._configured(), dict)

    def __call__(self, key: str) -> Any:
        configured = self._configured()
        if not isinstance(configured, dict):
            raise ImproperlyConfigured(f"{self.name} is a {type(configured).__name__}, not a dict.")
        default = self.defaults[key]
        value = configured.get(key, default)
        if isinstance(value, dict) and isinstance(default, dict):
            return {**default, **value}
        return value

    def errors(self, prefix: str) -> list[Error]:
        """One `<prefix>.E012` per problem."""
        configured = self._configured()
        if not isinstance(configured, dict):
            message = f"{self.name} is a {type(configured).__name__}, not a dict."
            return [Error(message, hint="Set a dict, or remove it.", id=f"{prefix}.E012")]
        try:
            self.model.model_validate(configured)
        except ValidationError as exc:
            return [
                Error(
                    f"{self.name}{''.join(f'[{part!r}]' for part in error['loc'] if part != '[key]')}: {error['msg']}.",
                    hint=f"The keys are {', '.join(self.defaults)}."
                    if error["type"] == "extra_forbidden"
                    else None,
                    id=f"{prefix}.E012",
                )
                for error in exc.errors()
            ]
        return []


setting = AppSettings("AI_SDK_OVERSIGHT", OversightSettings)
_retention = TypeAdapter(Retention)


def validate_retention(value: Any) -> dict[Any, int]:
    """Return `value` as retention days, or raise ValueError.

    The system check does not stop a turn or a prune, so runtime checks the same rule again.
    """
    try:
        return _retention.validate_python(value)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise ValueError(f"retention {list(error['loc'])}: {error['msg']}") from None


def deadline_ms(stage: str) -> float | None:
    """The deadline in milliseconds for `stage`, or None for no limit."""
    return (setting("DEADLINE_MS") or {}).get(str(stage))


def is_installed() -> bool:
    return apps.is_installed("django_ai_sdk.oversight")


def is_enabled() -> bool:
    """Whether the app is installed and ENABLED is true."""
    return is_installed() and bool(setting("ENABLED"))


__all__ = [
    "APP_LABEL",
    "deadline_ms",
    "is_enabled",
    "is_installed",
    "setting",
    "validate_retention",
]
