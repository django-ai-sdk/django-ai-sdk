import json


def format_sse(data: dict | str) -> bytes:
    """
    Format data as Server-Sent Event.

    Args:
        data: Dictionary to serialize as SSE data, or string for [DONE]

    Returns:
        Encoded SSE bytes
    """
    if isinstance(data, str):
        return f"data: {data}\n\n".encode()

    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()
