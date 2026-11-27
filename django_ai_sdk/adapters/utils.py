from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage


def merge_messages(
    messages: list[ChatMessage],
    roles: tuple[str, ...] = ("user", "assistant"),
    max_history: int | None = None,
) -> list[tuple[str, str]]:
    """max_history: Maximum number of most recent messages to keep (None = unlimited)"""
    filtered = [m for m in messages if m.role in roles]

    if max_history and len(filtered) > max_history:
        filtered = filtered[-max_history:]

    result: list[tuple[str, str]] = []
    for message in filtered:
        if result and result[-1][0] == message.role:
            result[-1] = (message.role, result[-1][1] + "\n\n" + message.content)
        else:
            result.append((message.role, message.content))

    return result
