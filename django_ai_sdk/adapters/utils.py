from django_ai_sdk.common import ChatMessage


def normalize_usage(usage_data: dict | None) -> dict | None:
    """
    Transform Haystack/OpenAI usage dict from snake_case to camelCase.

    Input:  {"prompt_tokens": 15, "completion_tokens": 36, "total_tokens": 51}
    Output: {"inputTokens": 15, "outputTokens": 36, "totalTokens": 51}

    Returns None if usage_data is empty/None.
    """
    if not usage_data:
        return None
    return {
        "inputTokens": usage_data.get("prompt_tokens", 0),
        "outputTokens": usage_data.get("completion_tokens", 0),
        "totalTokens": usage_data.get("total_tokens", 0),
    }


def merge_messages(
    messages: list[ChatMessage],
    roles: tuple[str, ...] = ("user", "assistant"),
    max_history: int | None = None,
) -> list[tuple[str, str]]:
    """
    Merge consecutive messages of the same role with history limiting.

    Args:
        messages: List of ChatMessage objects
        roles: Tuple of roles to include (default: ("user", "assistant"))
        max_history: Maximum number of most recent messages to keep (None = unlimited)

    Returns:
        List of (role, content) tuples with consecutive same-role messages merged

    Example:
        >>> messages = [
        ...     ChatMessage(role="user", content="Hello"),
        ...     ChatMessage(role="user", content="World"),
        ...     ChatMessage(role="assistant", content="Hi"),
        ... ]
        >>> merge_messages(messages)
        [("user", "Hello\n\nWorld"), ("assistant", "Hi")]
    """
    # Filter to specified roles
    filtered = [m for m in messages if m.role in roles]

    # Apply history limit
    if max_history and len(filtered) > max_history:
        filtered = filtered[-max_history:]

    # Merge consecutive messages of same role
    result: list[tuple[str, str]] = []
    for message in filtered:
        if result and result[-1][0] == message.role:
            result[-1] = (message.role, result[-1][1] + "\n\n" + message.content)
        else:
            result.append((message.role, message.content))

    return result
