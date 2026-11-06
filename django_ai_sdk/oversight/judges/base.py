"""Judges: models that answer a judge lens, and the shared machinery to ask them.

`ask` asks one judge with a timeout, so a slow or broken one becomes an error
answer instead of an exception. Lenses name judges by alias, and
AI_SDK_OVERSIGHT["JUDGES"] says what each alias is. A trailing "?" makes the
alias optional: a lens skips it when the deployment did not define it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.patterns import mask_secrets
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnContext

logger = get_logger(__name__)

# Seconds before an answer is abandoned, unless the judge sets its own.
DEFAULT_TIMEOUT = 30.0

# Error prefix for an answer whose external service failed.
# Reflect asks such a lens again later.
UNAVAILABLE = "unavailable"

# Error for text too long to judge whole. Nothing was judged, so FAIL_MODE decides.
TOO_LONG = "too long"

# Limits every judge family shares: earlier messages shown, characters per earlier
# message, and characters of a source title or tool name.
HISTORY_MESSAGES = 6
HISTORY_ITEM_CHARS = 500
TITLE_CHARS = 200


def modules_installed(*modules: str) -> bool:
    """Whether every one of `modules` can be imported, without importing it."""
    from importlib.util import find_spec

    return all(find_spec(module) is not None for module in modules)


def unavailable(exc: BaseException) -> str:
    """Error text for an external judge that raised `exc`."""
    return f"{UNAVAILABLE}: {type(exc).__name__}"


def is_unavailable(error: str) -> bool:
    """Whether `error` means the service failed and a later retry may work."""
    return error.startswith(f"{UNAVAILABLE}: ")


class Judge(ABC):
    """Answers lenses. Return None to abstain and let the next judge try."""

    # Who receives turn content, e.g. "model gateway". Empty means nothing leaves the process.
    external: ClassVar[str] = ""
    timeout: float = DEFAULT_TIMEOUT
    name: str = ""

    @abstractmethod
    async def answer(self, lens: Lens, view: TurnContext) -> Answer | None: ...

    def label(self) -> str:
        return self.name or type(self).__name__.lower()

    def missing(self) -> str:
        """Why this judge can't run here, or "" when it can. It abstains when set."""
        return ""

    def describe(self) -> dict[str, Any]:
        return {"name": self.label(), "external": self.external, "class": type(self).__name__}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.label()!r}>"


async def ask(judge: Judge | str | dict[str, Any], lens: Lens, view: TurnContext) -> Answer | None:
    """Ask one judge with a timeout. Any failure comes back as an Answer with `error`.

    `judge` may be a spec for `resolve`. A spec that fails to build is an error answer,
    so the lens keeps its other answers. An external judge that raises is marked
    unavailable. A task cancelled from outside re-raises; one that cancels itself is
    an error answer.
    """
    started = time.perf_counter()
    label = spec_label(judge)
    try:
        judge = resolve(judge)
    except Exception as exc:
        logger.warning("Judge {} for lens {} does not build: {}", label, lens.name, exc)
        return Answer(answerer=label, error=type(exc).__name__)

    def failed(error: str) -> Answer:
        return Answer(answerer=judge.label(), error=error, latency_ms=_ms(started))

    try:
        answer = await asyncio.wait_for(judge.answer(lens, view), timeout=judge.timeout)
    except TimeoutError:
        return failed("timeout")
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        logger.warning("Judge {} cancelled itself on lens {}", judge.label(), lens.name)
        return failed("CancelledError")
    except Exception as exc:
        if judge.external:
            # Log only the type. The message may quote the request or the reply.
            logger.warning(
                "Judge {} failed on lens {}: {}", judge.label(), lens.name, type(exc).__name__
            )
            return failed(unavailable(exc))
        logger.opt(exception=exc).warning("Judge {} failed on lens {}", judge.label(), lens.name)
        return failed(type(exc).__name__)
    if answer is None:
        return None
    if not isinstance(answer, Answer):
        logger.warning(
            "Judge {} answered lens {} with a {}, not an Answer",
            judge.label(),
            lens.name,
            type(answer).__name__,
        )
        return failed(f"not an answer: {type(answer).__name__}")
    if answer.value is not None and not answer.error:
        try:
            answer.value = lens.answer.parse(lens, answer.value)
        except Exception as exc:
            # Keep only the exception type. Its message may quote the value.
            answer = answer.model_copy(
                update={"value": None, "error": f"unparsable: {type(exc).__name__}"}
            )
    answer.answerer = answer.answerer or judge.label()
    if answer.latency_ms is None:
        answer.latency_ms = _ms(started)
    return answer


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def is_optional(spec: Any) -> bool:
    """Whether `spec` is an optional alias like "judge?"."""
    return isinstance(spec, str) and spec.endswith("?")


def alias_of(spec: Any) -> str:
    """The alias a judge spec names, without the optional "?"."""
    return str(spec).removesuffix("?")


def spec_label(spec: Any) -> str:
    """A name for a judge spec, for logs and checks, without building it."""
    if isinstance(spec, Judge):
        return spec.label()
    if isinstance(spec, dict):
        return str(spec.get("name") or spec.get("class") or "judge")
    return str(spec)


def clip(text: str, limit: int) -> str:
    """Shorten `text` to `limit` characters by cutting out the middle.

    Keep both ends because an injected instruction can hide after a page of padding.
    """
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters left out ...]\n"
    keep = max(limit - len(marker.format(len(text))), 0)
    head, tail = keep - keep // 2, keep // 2
    return text[:head] + marker.format(len(text) - keep) + text[len(text) - tail :]


def masked_clip(text: str, limit: int) -> str:
    """Mask secrets, then clip to `limit`. Every judge reads turn text this way."""
    return clip(mask_secrets(text or ""), limit)


def too_long(
    lens: Lens, view: TurnContext, *, input_chars: int, output_chars: int, tool_chars: int
) -> bool:
    """Whether the text `lens` judges would have to be clipped. If so, nothing is judged.

    Only the judged text counts (the question, the answer, or tool arguments).
    Context like history and sources is clipped to fit instead.
    """
    judged: list[tuple[str, int]] = []
    if lens.subject in {Subject.INPUT, Subject.EXCHANGE}:
        judged.append((view.input, input_chars))
    per_source = lens.answer.kind == "entities" and lens.per_source
    if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE} or per_source:
        judged.append((view.output, output_chars))
    if lens.subject is Subject.TOOL_CALL and view.tool_call:
        arguments = json.dumps(view.tool_call.arguments, ensure_ascii=False, default=str)
        judged.append((arguments, tool_chars))
    return any(len(text) > limit and len(mask_secrets(text)) > limit for text, limit in judged)


def unit(value: Any, default: float) -> float:
    """`value` as a number from 0 to 1, or `default` when it is not a number."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def secret_setting(name: str) -> str:
    """A key named `name`, from the environment or else Django settings. "" when unset."""
    from django_ai_sdk.utils import resolve_setting

    return str(os.environ.get(name) or resolve_setting(name, "") or "").strip()


def sampled(key: str, rate: float) -> bool:
    """Deterministic sampling: the same `key` always gets the same result for a `rate`."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def resolve(spec: Judge | str | dict[str, Any]) -> Judge:
    """Build a Judge from an instance, an alias, or a config dict.

    A config dict is {"class": dotted path, **kwargs}. An alias may point to another
    alias, in `alias_table()`.
    """
    if isinstance(spec, Judge):
        return spec
    if isinstance(spec, str):
        return _build_alias(alias_of(spec), alias_table())
    if isinstance(spec, dict):
        return _from_config(spec)
    raise ImproperlyConfigured(f"{spec!r} is not a judge, an alias or a config")


def alias_table() -> dict[str, Any]:
    """All aliases from AI_SDK_OVERSIGHT["JUDGES"]."""
    from django_ai_sdk.oversight.conf import setting

    return dict(setting("JUDGES"))


def alias_defined(alias: str) -> bool:
    return alias in alias_table()


def _build_alias(
    alias: str, aliases: Mapping[str, Any], seen: frozenset[str] = frozenset()
) -> Judge:
    if alias in seen:
        raise ImproperlyConfigured(f"Judge alias {alias!r} names itself")
    if alias not in aliases:
        raise ImproperlyConfigured(
            f"No judge alias {alias!r}; add it to AI_SDK_OVERSIGHT['JUDGES']"
        )
    target = aliases[alias]
    if isinstance(target, str):
        return _build_alias(target, aliases, seen | {alias})
    judge = resolve(target)
    if not (isinstance(target, dict) and target.get("name")):
        # Name it after the alias lenses use. Each answer records the model anyway.
        judge.name = alias
    return judge


def _from_config(config: dict[str, Any]) -> Judge:
    config = dict(config)
    path = config.pop("class", "")
    if not path:
        raise ImproperlyConfigured(f"Judge config {config!r} has no 'class'")
    if isinstance(path, str) and "." not in path:
        # A bare name like "LLMJudge" means one of the SDK's own judges.
        path = f"django_ai_sdk.oversight.judges.{path}"
    cls = import_string(path) if isinstance(path, str) else path
    if not (isinstance(cls, type) and issubclass(cls, Judge)):
        raise ImproperlyConfigured(f"{path!r} is not a Judge subclass")
    return cls(**config)


__all__ = [
    "DEFAULT_TIMEOUT",
    "HISTORY_ITEM_CHARS",
    "HISTORY_MESSAGES",
    "TITLE_CHARS",
    "TOO_LONG",
    "UNAVAILABLE",
    "Judge",
    "alias_defined",
    "alias_of",
    "alias_table",
    "ask",
    "clip",
    "is_optional",
    "is_unavailable",
    "masked_clip",
    "modules_installed",
    "resolve",
    "sampled",
    "secret_setting",
    "spec_label",
    "too_long",
    "unavailable",
    "unit",
]
