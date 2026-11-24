"""The Judge base and the shared machinery to ask one, build one from an alias or config."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.calls import settled
from django_ai_sdk.oversight.conf import build_configured, setting
from django_ai_sdk.oversight.patterns import mask_secrets
from django_ai_sdk.oversight.types import Answer, Subject
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnContext

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0

# Error prefix for a failed external service; reflect asks such a lens again later.
UNAVAILABLE = "unavailable"

# Errors where nothing was judged, so FAIL_MODE decides.
TOO_LONG = "too long"
NO_VERDICT = "no verdict"

# Below the default min_confidence, so a made-up confidence never ends the chain.
UNSTATED_CONFIDENCE = 0.5

HISTORY_MESSAGES = 6
HISTORY_ITEM_CHARS = 500
TITLE_CHARS = 200


def modules_installed(*modules: str) -> bool:
    return all(find_spec(module) is not None for module in modules)


def unavailable(exc: BaseException) -> str:
    return f"{UNAVAILABLE}: {type(exc).__name__}"


def is_unavailable(error: str) -> bool:
    return error.startswith(f"{UNAVAILABLE}: ")


class Judge(ABC):
    """Answers lenses. Return None to abstain and let the next judge try.

    `confidence` is 0 to 1 on this judge's own scale, compared with a lens's
    `min_confidence`: report what you can stand behind, or it ends the chain every
    time. A judge that is a model names it in `model`.
    """

    # Who receives turn content, e.g. "model gateway". Empty means nothing leaves the process.
    external: ClassVar[str] = ""
    timeout: float = DEFAULT_TIMEOUT
    name: str = ""
    # The most characters of one text it reads; longer is TOO_LONG. 0: no limit.
    TEXT_CHARS: ClassVar[int] = 0

    def __init__(self, *, name: str = "", timeout: float | None = None) -> None:
        if name:
            self.name = name
        if timeout is not None:
            self.timeout = timeout

    @abstractmethod
    async def answer(self, lens: Lens, context: TurnContext) -> Answer | None: ...

    @property
    def display_name(self) -> str:
        return self.name or type(self).__name__.lower()

    def missing(self) -> str:
        """Why this judge can't run here (e.g. an unset key), or "" when it can."""
        return ""

    def answers(self, lens: Lens) -> bool:
        """Whether this judge can answer `lens` at all, e.g. its answer type."""
        return True

    def describe(self) -> dict[str, Any]:
        return {"name": self.display_name, "external": self.external, "class": type(self).__name__}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.display_name!r}>"


async def ask_judge(
    judge: Judge | str | dict[str, Any], lens: Lens, context: TurnContext
) -> Answer | None:
    """Ask one judge with a timeout. Any failure comes back as an Answer with `error`.

    An answer without a value or an error is NO_VERDICT, so FAIL_MODE applies
    (return None to abstain). Cancellation from outside re-raises.
    """
    started = time.perf_counter()
    label = judge_label(judge)
    try:
        judge = build_judge(judge)
        if judge.missing():
            return None
    except Exception as exc:
        logger.warning("Judge {} for lens {} does not build: {}", label, lens.name, exc)
        return Answer(answerer=label, error=type(exc).__name__)

    def failed(error: str) -> Answer:
        return Answer(
            answerer=judge.display_name,
            error=error,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    try:
        # A judge's own model calls are oversight, not the turn's.
        with settled():
            answer = await asyncio.wait_for(judge.answer(lens, context), timeout=judge.timeout)
    except TimeoutError:
        return failed("timeout")
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        logger.warning("Judge {} cancelled itself on lens {}", judge.display_name, lens.name)
        return failed("CancelledError")
    except Exception as exc:
        if judge.external:
            # Log only the type. The message may quote the request or the reply.
            logger.warning(
                "Judge {} failed on lens {}: {}", judge.display_name, lens.name, type(exc).__name__
            )
            return failed(unavailable(exc))
        logger.opt(exception=exc).warning(
            "Judge {} failed on lens {}", judge.display_name, lens.name
        )
        return failed(type(exc).__name__)
    if answer is None:
        return None
    if not isinstance(answer, Answer):
        logger.warning(
            "Judge {} answered lens {} with a {}, not an Answer",
            judge.display_name,
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
    elif answer.value is None and not answer.error:
        answer = answer.model_copy(update={"error": NO_VERDICT})
    answer.answerer = answer.answerer or judge.display_name
    if answer.latency_ms is None:
        answer.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return answer


@dataclass(frozen=True)
class Maybe:
    """A judge alias asked only when JUDGES defines it. JSON: {"alias": ..., "optional": true}."""

    alias: str


def judge_from_json(spec: Any) -> Any:
    if not (isinstance(spec, dict) and "alias" in spec):
        return spec
    alias = spec["alias"]
    if set(spec) != {"alias", "optional"} or spec["optional"] is not True or not alias:
        raise ImproperlyConfigured(
            f'Judge {spec!r}: an optional alias is {{"alias": "name", "optional": true}}'
        )
    return Maybe(str(alias))


def judge_label(spec: Any) -> str:
    if isinstance(spec, Judge):
        return spec.display_name
    if isinstance(spec, dict):
        return str(spec.get("name") or spec.get("class") or "judge")
    return str(spec)


def clip(text: str, limit: int) -> str:
    """Cut out the middle: an injected instruction can hide after a page of padding."""
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters left out ...]\n"
    keep = max(limit - len(marker.format(len(text))), 0)
    head, tail = keep - keep // 2, keep // 2
    return text[:head] + marker.format(len(text) - keep) + text[len(text) - tail :]


def masked_clip(text: str, limit: int) -> str:
    """Every judge reads turn text this way: secrets masked, then clipped."""
    return clip(mask_secrets(text or ""), limit)


def too_long(
    lens: Lens, context: TurnContext, *, input_chars: int, output_chars: int, tool_chars: int
) -> bool:
    """Whether the judged text would need clipping; then nothing is judged.

    Context like history and sources is clipped to fit instead.
    """
    judged: list[tuple[str, int]] = []
    if lens.subject in {Subject.QUESTION, Subject.EXCHANGE}:
        judged.append((context.input, input_chars))
    per_source = lens.answer.kind == "entities" and lens.per_source
    if lens.subject in {Subject.ANSWER, Subject.EXCHANGE} or per_source:
        judged.append((context.output, output_chars))
    if lens.subject is Subject.TOOL_CALL and context.tool_call:
        arguments = json.dumps(context.tool_call.arguments, ensure_ascii=False, default=str)
        judged.append((arguments, tool_chars))
    return any(len(text) > limit and len(mask_secrets(text)) > limit for text, limit in judged)


def unit(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def secret_setting(name: str) -> str:
    """From the environment, else Django settings."""
    return str(os.environ.get(name) or resolve_setting(name, "") or "").strip()


def sampled(key: str, rate: float) -> bool:
    """Deterministic: the same `key` always gets the same result for a `rate`."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def build_judge(spec: Judge | str | dict[str, Any]) -> Judge:
    """Build a Judge from an instance, an alias, or {"class": dotted path, **kwargs}."""
    if isinstance(spec, Judge):
        return spec
    if isinstance(spec, str):
        return _build_alias(spec, alias_table())
    if isinstance(spec, dict):
        return build_configured(spec, Judge, "django_ai_sdk.oversight.judges")
    raise ImproperlyConfigured(f"{spec!r} is not a judge, an alias or a config")


def alias_table() -> dict[str, Any]:
    return dict(setting("JUDGES"))


def alias_defined(alias: str) -> bool:
    return alias in alias_table()


def _build_alias(alias: str, aliases: Mapping[str, Any]) -> Judge:
    target: Any = alias
    seen: set[str] = set()
    while isinstance(target, str):
        if target in seen:
            raise ImproperlyConfigured(f"Judge alias {target!r} names itself")
        if target not in aliases:
            raise ImproperlyConfigured(
                f"No judge alias {target!r}; add it to AI_SDK_OVERSIGHT['JUDGES']"
            )
        seen.add(target)
        target = aliases[target]
    # Copy a shared instance, so renaming it doesn't rename it for other aliases.
    judge = copy.copy(target) if isinstance(target, Judge) else build_judge(target)
    if not (isinstance(target, dict) and target.get("name")):
        # Name it after the alias the lens used. Each answer records the model anyway.
        judge.name = alias
    return judge
