from __future__ import annotations

import json


def format_sse(data: dict | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode()

    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()
