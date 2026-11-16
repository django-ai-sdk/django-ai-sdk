"""TurnRequest, carried in the run's `hook_context` so hooks fill in the caller's object."""

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
    """A run's turn: what the caller wants of it, then what oversight decided.

    Pass it to `run_with_oversight(turn=...)`. Once the run started, `guard` is the
    turn it opened, or None when it joined an outer turn or has no policy.
    """

    # The SDK agent whose policy the run is under.
    agent: Any = None
    # Such as "api:extract"; the app picks its own names. None records "default".
    channel: str | None = None
    user: Any = None
    thread_id: str | None = None
    # A policy for this run only, as in `Agent.run(oversight=...)`.
    oversight: type[Policy] | str | Literal[False] | None = None
    # The stored thread vouches for the history, so the gate does not judge it again.
    history_vouched: bool = False
    # Called once the gate passed the question, such as to store it.
    on_gated: Callable[[TurnRequest], Awaitable[None]] | None = None
    # Reads the answer finish judges off the produced messages (e.g. a subagent's);
    # a replacement then takes every produced message's place. Default: all model
    # text, and a replacement takes the last reply's place.
    read_answer: Callable[[list[HaystackChatMessage]], str] | None = None
    # A Pydantic model the answer must still fit once masked.
    answer_schema: type[BaseModel] | None = None
    # The id of the message that will store the answer.
    message_id: str = ""
    # The caller closes the turn itself, such as a stream that still stores the answer.
    keep_open: bool = False
    # Called with the stubs of blocked tool calls, which never reach the ToolInvoker.
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
