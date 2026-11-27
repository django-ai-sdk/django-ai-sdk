from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.agents import declared_policy
from django_ai_sdk.oversight.conf import is_enabled
from django_ai_sdk.oversight.guard.masking import gate_masking_lenses
from django_ai_sdk.oversight.registry import get_resolved_policy, policy_name

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
    channel: str | None = None,
    user: Any = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> tuple[list[ChatMessage], bool]:
    """The messages a turn sends, and whether the stored thread vouches for the history."""
    if storage is None:
        return messages, False
    policy = await policy_for(agent, channel=channel, user=user, oversight=oversight)
    vouched = await thread_history(policy, storage, messages)
    return (vouched, True) if vouched is not None else (messages, False)


def takes_stored_history(policy: ResolvedPolicy | None) -> bool:
    return policy is not None and bool(gate_masking_lenses(policy))


async def thread_history(
    policy: ResolvedPolicy | None, storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage] | None:
    if takes_stored_history(policy):
        return await stored_history(storage_adapter, messages)
    if await _as_stored(storage_adapter, messages):
        return messages
    return None


async def policy_for(
    agent: Any,
    *,
    channel: str | None = None,
    user: Any = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> ResolvedPolicy | None:
    if oversight in (False, "") or not is_enabled():
        return None
    try:
        name = policy_name(oversight) if oversight else ""
        name = name or (await declared_policy(agent, channel=channel, user=user))[0]
        return await get_resolved_policy(name) if name else None
    except Exception:
        return None


async def stored_history(
    storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage]:
    earlier, newest = _split_question(messages)
    kept = await _stored_for(storage_adapter, earlier)
    return [*kept, newest] if newest is not None else kept


async def thread_storage(agent: Any, thread_id: str | None) -> BaseStorageAdapter | None:
    get_storage = getattr(agent, "get_storage_adapter", None)
    if not thread_id or get_storage is None:
        return None
    try:
        storage = await get_storage(thread_id)
        if storage is not None and await type(storage).get_thread(thread_id):
            return storage
    except Exception:
        logger.opt(exception=True).debug("No stored thread {}", thread_id)
    return None


async def _as_stored(storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]) -> bool:
    earlier, _ = _split_question(messages)
    if not earlier:
        return True
    kept = await _stored_for(storage_adapter, earlier)
    return len(kept) == len(earlier) and all(
        k.role == m.role and (k.content or "") == (m.content or "")
        for k, m in zip(kept, earlier, strict=True)
    )


def _split_question(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], ChatMessage | None]:
    if messages and messages[-1].role == "user":
        return messages[:-1], messages[-1]
    return list(messages), None


async def _stored_for(
    storage_adapter: BaseStorageAdapter, earlier: list[ChatMessage]
) -> list[ChatMessage]:
    if not earlier:
        return []
    stored = [m for m in await storage_adapter.get_messages() if m.role in ("user", "assistant")]
    by_id = {m.id: m for m in stored if m.id}
    if all(m.id and m.id in by_id for m in earlier):
        return [by_id[m.id] for m in earlier]
    return stored[: len(earlier)]
