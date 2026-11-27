from __future__ import annotations

import contextvars
import functools
import inspect
import types
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import is_enabled, setting
from django_ai_sdk.utils import scoped

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

logger = get_logger(__name__)

_required: contextvars.ContextVar[str] = contextvars.ContextVar(
    "django_ai_sdk_oversight_turn_required", default=""
)
_settled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_settled", default=False
)
_warned: set[str] = set()


class UnguardedModelCall(RuntimeError):
    """A model was called outside a required turn while UNGUARDED_MODEL_CALLS is "refuse"."""


def requires_turn(where: str) -> AbstractContextManager[None]:
    return scoped(_required, where)


def settled() -> AbstractContextManager[None]:
    return scoped(_settled, True)


def _check_model_call(generator: Any) -> None:
    where = _required.get()
    if not where or _settled.get() or not is_enabled():
        return
    from django_ai_sdk.oversight.guard import current_guard, is_unguarded

    guard = current_guard()
    if is_unguarded() or (guard is not None and guard.gated):
        return
    mode = setting("UNGUARDED_MODEL_CALLS")
    if mode == "allow":
        return
    name = type(generator).__name__
    message = (
        f"{name} was called in {where} outside an oversight turn, so no policy saw it. "
        "Call the model through Agent.run() or turn() (django_ai_sdk.oversight.turns)."
    )
    if mode == "refuse":
        raise UnguardedModelCall(message)
    if where not in _warned:
        _warned.add(where)
        logger.warning(message)


def watched[T](generator: T) -> T:
    """Bound on the instance, so Haystack's deep copies stay watched and class signatures intact."""
    cls = type(generator)
    for name in ("run", "run_async"):
        method = getattr(cls, name, None)
        if method is None:
            continue
        setattr(generator, name, types.MethodType(_checked(method), generator))
    return generator


def watched_factory[**P, T](factory: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(factory)
    def build(*args: P.args, **kwargs: P.kwargs) -> T:
        return watched(factory(*args, **kwargs))

    return build


def _checked(method: Any) -> Any:
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def run_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            _check_model_call(self)
            return await method(self, *args, **kwargs)

        return run_async

    @functools.wraps(method)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        _check_model_call(self)
        return method(self, *args, **kwargs)

    return run


__all__ = [
    "UnguardedModelCall",
    "requires_turn",
    "settled",
    "watched",
    "watched_factory",
]
