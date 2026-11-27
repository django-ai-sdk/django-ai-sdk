from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy as _

from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.content import stored_value
from django_ai_sdk.oversight.notifiers import notify_events
from django_ai_sdk.oversight.rules import ACTION_ORDER, USER_FACING, Block, RestrictTools
from django_ai_sdk.oversight.types import Outcome, Stage, worst_outcome

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.judging import Matched
    from django_ai_sdk.oversight.guard.stages import StageResult

TOO_LONG_MESSAGE = _("This message is too long to check. Shorten it and try again.")
UNCHECKED_TEXT = _("[withheld: could not be checked]")
BLOCKED_TEXT = _("[withheld: blocked by oversight]")
TOOL_UNAVAILABLE_MESSAGE = _("This tool is not available for this request.")


def notice_base(turn_id: Any, stage: Stage, outcome: Outcome) -> dict[str, Any]:
    return {"turn_id": str(turn_id), "stage": str(stage), "outcome": str(outcome)}


def rule_notes(matched: list[Matched], told: set[str]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for match in matched:
        rule = match.rule
        if not match.acted or not rule.notice or match.failed or rule.name in told:
            continue
        told.add(rule.name)
        acted = {a.kind for a in match.acted}
        kinds = [k for k in ACTION_ORDER if k in USER_FACING and k in acted]
        notes.append(
            {
                "kind": kinds[0] if kinds else "notice",
                "also": kinds[1:],
                "text": str(rule.notice),
                "stage": str(match.stage),
            }
        )
    return notes


class TurnNotices:
    def __init__(self, guard: TurnGuard) -> None:
        self.guard = guard
        self.pending: list[dict[str, Any]] = []
        self.told: set[str] = set()

    def stage_notice(
        self,
        stage: Stage,
        matched: list[Matched],
        blocked: bool,
        message: str,
        finished: StageResult | None = None,
    ) -> dict[str, Any] | None:
        acted = [m for m in matched if m.acted]
        level = setting("CLIENT_DETAIL")
        detail = level == "full"
        notes = rule_notes([*acted, *self._tool_matches(stage, matched)], self.told)
        shown = [
            m
            for m in acted
            if detail
            or (m.rule.notice and not m.failed)
            or any(a.kind in USER_FACING for a in m.acted)
        ]
        if not shown and not detail and not notes:
            return None
        notice = notice_base(self.guard.id, stage, worst_outcome(*(m.outcome() for m in shown)))
        if notes:
            notice["notes"] = notes
        if blocked:
            notice["message"] = message
        if finished is not None and finished.changed:
            notice["replace"] = finished.text
        masked = any(a.kind == "mask" for m in acted for a in m.acted)
        if stage is Stage.GATE and masked and not blocked:
            notice["input"] = self.guard.context.input
        if detail:
            notice["rules"] = [m.rule.name for m in acted]
            notice["detail"] = self._detail(stage, matched)
        return notice

    def _detail(self, stage: Stage, matched: list[Matched]) -> dict[str, Any]:
        policy = self.guard.resolved
        matches = [*matched, *self._tool_matches(stage, matched)]
        return {
            "policy": policy.name,
            "report_outcome": str(worst_outcome(*(m.outcome(report=True) for m in matches))),
            "pending": sorted(lens.name for lens in policy.active_lenses(Stage.REFLECT))
            if stage is Stage.FINISH
            else [],
            "answers": [
                {
                    "lens": o.lens.name,
                    "label": o.lens.display_name,
                    "kind": o.lens.answer.kind,
                    "value": stored_value(o.lens, o.answer),
                    "confidence": o.answer.confidence if o.answer.ok else None,
                    "answerer": o.answer.answerer,
                    "model": o.answer.model,
                    "error": o.answer.error,
                    "latency_ms": o.answer.latency_ms,
                }
                for o in self.guard.answered
                if o.stage is stage
            ],
            "matches": [
                {
                    "rule": m.rule.name,
                    "mode": str(m.rule.mode),
                    "acted": [a.kind for a in m.acted],
                    "would": [a.kind for a in m.rule.then],
                    "notice": str(m.rule.notice),
                }
                for m in matches
            ],
        }

    def _tool_matches(self, stage: Stage, matched: list[Matched]) -> list[Matched]:
        if stage is not Stage.FINISH:
            return []
        return [m for m in self.guard.matched if m.stage is Stage.TOOL and m not in matched]

    def notifications(self, matched: list[Matched]) -> list[tuple[dict[str, Any], list[str]]]:
        return notify_events(
            self.guard.id,
            matched,
            thread_id=self.guard.thread_id,
            agent=self.guard.context.agent_name,
            channel=self.guard.channel,
            outcome=str(self.guard.outcome),
        )


def first_block_message(matched: Iterable[Matched]) -> str | None:
    for match in matched:
        for action in match.acted:
            if isinstance(action, Block):
                return block_message(match, action)
    return None


def restricted_tools(matched: Iterable[Matched], restrict: list[str] | None) -> list[str] | None:
    for match in matched:
        for action in match.acted:
            if isinstance(action, RestrictTools):
                restrict = merge_restrict(restrict, action.names)
    return restrict


def merge_restrict(restrict: list[str] | None, names: list[str]) -> list[str]:
    if not names or restrict == []:
        return []
    return sorted({*(restrict or []), *names})


def block_message(match: Matched, block: Block) -> str:
    if match.failed:
        return str(Block.DEFAULT_MESSAGE)
    return str(match.rule.message or block.message)


def removed_tool_message(match: Matched) -> str:
    if match.failed:
        return str(TOOL_UNAVAILABLE_MESSAGE)
    return str(match.rule.message or TOOL_UNAVAILABLE_MESSAGE)
