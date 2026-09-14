from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.workflows.actions import ActionContext


class ConsoleLogAction:
    description = "Print workflow output to console (dev/debug)"

    async def execute(self, payload: Any, context: ActionContext) -> None:
        print(f"[{context.source}] {json.dumps(payload, indent=2, default=str)}")  # noqa: T201
