"""Seeding a workflow run's inputs bag.

Messages are always JSON-safe dicts, since the bag is written to a JSONField.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage


def dump_message(item: Any) -> dict[str, Any]:
    """One message as a JSON-safe dict."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return {"role": "user", "content": str(item)}


def dump_messages(raw: Any) -> list[dict[str, Any]]:
    """A list of messages as JSON-safe dicts."""
    return [dump_message(item) for item in (raw or [])]


def normalize_workflow_inputs(
    *,
    inputs: dict[str, Any] | None = None,
    messages: list[ChatMessage] | list[dict[str, Any]] | None = None,
    ensure_messages: bool = False,
) -> dict[str, Any]:
    """Merge caller inputs with optional messages into one bag.

    Explicit `messages=` wins over `inputs["messages"]`. `ensure_messages` adds an
    empty list when neither supplies one, so agent steps always find the key.
    """
    bag = dict(inputs or {})
    if messages is not None:
        bag["messages"] = dump_messages(messages)
    elif "messages" in bag:
        bag["messages"] = dump_messages(bag["messages"])
    elif ensure_messages:
        bag["messages"] = []
    return bag


__all__ = ["dump_message", "dump_messages", "normalize_workflow_inputs"]
