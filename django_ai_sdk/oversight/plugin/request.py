from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from haystack.dataclasses import ChatMessage as HaystackChatMessage
    from pydantic import BaseModel

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.stages import StageResult
    from django_ai_sdk.oversight.policy import Policy

REQUEST_KEY = "django_ai_sdk.oversight"


@dataclass
class TurnRequest:
    """Once the run started, `guard` is the turn it opened, or None when it joined an
    outer turn or has no policy.
    """

    agent: Any = None
    channel: str | None = None
    user: Any = None
    thread_id: str | None = None
    oversight: type[Policy] | str | Literal[False] | None = None
    history_vouched: bool = False
    on_gated: Callable[[TurnRequest], Awaitable[None]] | None = None
    read_answer: Callable[[list[HaystackChatMessage]], str] | None = None
    answer_schema: type[BaseModel] | None = None
    message_id: str = ""
    keep_open: bool = False
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    guard: TurnGuard | None = None
    gate_result: StageResult | None = None
    first_produced_index: int = 0

    @property
    def question_blocked(self) -> bool:
        return self.gate_result is not None and self.gate_result.blocked

    @property
    def finish_result(self) -> StageResult | None:
        return self.guard.finish_result if self.guard is not None else None

    def raise_if_blocked(self) -> None:
        """Raise `Blocked` when the gate blocked the question or finish withheld the answer."""
        if self.guard is None:
            return
        if self.question_blocked:
            assert self.gate_result is not None
            raise Blocked(self.gate_result.message, stage=str(Stage.GATE), turn_id=self.guard.id)
        if (finish := self.finish_result) is not None and finish.blocked:
            raise Blocked(finish.message, stage=str(Stage.FINISH), turn_id=self.guard.id)

    def hook_context(self, **extra: Any) -> dict[str, Any]:
        return {**extra, REQUEST_KEY: self}
