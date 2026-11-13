"""Which history a turn judges: the client's copy, or the stored thread that vouches for it.

`Agent.run` and chat both decide it here, before the turn opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.agents import acode_policy
from django_ai_sdk.oversight.guard.masks import gate_masking_lenses

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = get_logger(__name__)


async def history_for(
    agent: Any,
    storage: BaseStorageAdapter | None,
    messages: list[ChatMessage],
    *,
    channel: str,
    user: Any = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> tuple[list[ChatMessage], bool]:
    """The messages a turn of `agent` sends, and whether the stored thread vouches for the history.

    `storage` is the stored thread (see `thread_storage`), or None when there is none.
    """
    if storage is None:
        return messages, False
    policy = await resolve_policy(agent, channel=channel, user=user, oversight=oversight)
    vouched = await thread_history(policy, storage, messages)
    return (vouched, True) if vouched is not None else (messages, False)


def takes_stored_history(policy: ResolvedPolicy | None) -> bool:
    """Whether the turn uses the stored thread as its history.

    Only when the policy enforces masking at the gate. Then the stored copy is
    the masked one, and each message in it was judged in its own turn. Otherwise
    the client's history is used and judged by `gate_history`, so a policy in
    report mode changes nothing.
    """
    return policy is not None and bool(gate_masking_lenses(policy))


async def thread_history(
    policy: ResolvedPolicy | None, storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage] | None:
    """Return the history the stored thread vouches for, or None if it must be judged.

    Under a policy that masks at the gate, the stored messages replace the sent
    ones (see `stored_history`). Otherwise a sent history that matches the stored
    thread (role and text) was already judged and is kept. One that differs was
    edited or made up, so it gets judged.
    """
    if takes_stored_history(policy):
        return await stored_history(storage_adapter, messages)
    if await _as_stored(storage_adapter, messages):
        return messages
    return None


async def resolve_policy(
    agent: Any,
    *,
    channel: str,
    user: Any = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> ResolvedPolicy | None:
    """The policy a turn of `agent` would run under, or None. Never raises.

    For choosing the history before the run opens its turn. None (no policy, or
    it failed to load) means the history is judged as sent, which is the safe side.
    """
    from django_ai_sdk.oversight.conf import is_enabled
    from django_ai_sdk.oversight.registry import policy_name
    from django_ai_sdk.oversight.resolve import aget_policy

    if oversight in (False, "") or not is_enabled():
        return None
    try:
        name = policy_name(oversight) if oversight else ""
        name = name or (await acode_policy(agent, channel=channel, user=user))[0]
        return await aget_policy(name) if name else None
    except Exception:
        return None


async def stored_history(
    storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage]:
    """Replace the history in `messages` with the stored thread. Keep the new question.

    The client's copy may be unmasked or made up, so it is not trusted. When every
    earlier message has a stored id, those stored messages are used. Otherwise the
    stored messages are taken up to the length of the client's history, so a
    regenerated or edited turn still drops what came after. Only user and
    assistant messages count.
    """
    earlier, newest = _split_question(messages)
    kept = await _stored_for(storage_adapter, earlier)
    return [*kept, newest] if newest is not None else kept


async def thread_storage(agent: Any, thread_id: str | None) -> BaseStorageAdapter | None:
    """Return the storage for `thread_id`, or None when the thread is not stored."""
    get_storage = getattr(agent, "get_storage_adapter", None)
    if not thread_id or get_storage is None:
        return None
    try:
        storage = await get_storage(thread_id)
        if storage is not None and await type(storage).get_thread(thread_id):
            return storage
    except Exception:
        # Without a stored thread, the history is judged as sent.
        logger.opt(exception=True).debug("No stored thread {}", thread_id)
    return None


async def _as_stored(storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]) -> bool:
    """Whether every message before the question matches the stored thread."""
    earlier, _ = _split_question(messages)
    if not earlier:
        return True
    kept = await _stored_for(storage_adapter, earlier)
    # A shorter match means the client sent more history than the thread holds.
    return len(kept) == len(earlier) and all(
        k.role == m.role and (k.content or "") == (m.content or "")
        for k, m in zip(kept, earlier, strict=True)
    )


def _split_question(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], ChatMessage | None]:
    """Split `messages` into the history and the new question, if the last one is a question."""
    if messages and messages[-1].role == "user":
        return messages[:-1], messages[-1]
    return list(messages), None


async def _stored_for(
    storage_adapter: BaseStorageAdapter, earlier: list[ChatMessage]
) -> list[ChatMessage]:
    """The stored messages that stand for `earlier`: by id, or else by position.

    By position it may return fewer messages than `earlier` when the thread is shorter.
    """
    if not earlier:
        return []
    stored = [m for m in await storage_adapter.get_messages() if m.role in ("user", "assistant")]
    by_id = {m.id: m for m in stored if m.id}
    if all(m.id and m.id in by_id for m in earlier):
        return [by_id[m.id] for m in earlier]
    return stored[: len(earlier)]


__all__ = [
    "history_for",
    "resolve_policy",
    "stored_history",
    "takes_stored_history",
    "thread_history",
    "thread_storage",
]
