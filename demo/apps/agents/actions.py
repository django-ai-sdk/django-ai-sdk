from __future__ import annotations

import json
from typing import Any


class ConsoleLogAction:
    description = "Print workflow output to console (dev/debug)"

    async def execute(self, payload: Any) -> None:
        print(json.dumps(payload, indent=2, default=str))  # noqa: T201
