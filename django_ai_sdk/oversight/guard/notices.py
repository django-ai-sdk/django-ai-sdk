"""Refusal messages, client notices and notifier events for a turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy as _

from django_ai_sdk.oversight.actions import ACTION_ORDER, USER_FACING, Block, RestrictTools
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.guard.state import GuardState, merge_restrict
from django_ai_sdk.oversight.types import Outcome, Stage, worst_outcome

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard.asking import Matched
    from django_ai_sdk.oversight.guard.finish import FinishResult

TOO_LONG_MESSAGE = _("This message is too long to check. Shorten it and try again.")
# Shown to the model instead of a tool result or earlier message that could not be checked.
UNCHECKED_TEXT = _("[withheld: could not be checked]")
# Shown to the model instead of a tool result an enforced gate rule refuses.
REFUSED_TEXT = _("[withheld: refused by oversight]")
# The refusal for a call to a tool removed this turn.
TOOL_UNAVAILABLE = _("This tool is not available for this request.")


def notice_base(turn_id: Any, stage: Stage, outcome: Outcome) -> dict[str, Any]:
    """Return the fields every `data-oversight` payload starts with."""
    return {"turn_id": str(turn_id), "stage": str(stage), "outcome": str(outcome)}


def rule_notes(matched: list[Matched], told: set[str]) -> list[dict[str, Any]]:
    """Build the notes shown to the person for rules that acted, each rule once per turn.

    `kind` is the first visible (USER_FACING) action by ACTION_ORDER, for the
    client's icon, and `also` lists the rest. When the rule acted only out of
    sight (a queue, alert or notification), `kind` is "notice" so the client
    never learns which. `stage` is where the rule acted.
    """
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


class Notices(GuardState):
    """Builds a stage's `data-oversight` payloads and the turn's Notify events."""

    def _notice(
        self,
        stage: Stage,
        matched: list[Matched],
        blocked: bool,
        message: str,
        finished: FinishResult | None = None,
    ) -> dict[str, Any] | None:
        acted = [m for m in matched if m.acted]
        level = setting("CLIENT_DETAIL")
        detail = level == "full"
        # Tool calls send no notice of their own, so the answer's notice covers them.
        notes = rule_notes([*acted, *self._tool_matches(stage, matched)], self._told)
        # Hide invisible actions (queue, alert, notify) so the client cannot probe the
        # policy. A rule's own notice may still reveal them.
        shown = acted if detail else [m for m in acted if _seen(m)]
        if not shown and not detail and not notes:
            return None
        notice = notice_base(self.id, stage, worst_outcome(*(m.outcome() for m in shown)))
        if notes:
            notice["notes"] = notes
        if blocked:
            notice["message"] = message
        if finished is not None and finished.replaced:
            notice["replace"] = finished.output
        # Never echo a refused question, even masked, since parts may be unsafe.
        masked = any(a.kind == "mask" for m in acted for a in m.acted)
        if stage is Stage.GATE and masked and not blocked:
            notice["input"] = self.context.input
        if detail:
            # Rule names reveal what the policy tests, so only send them in full detail.
            notice["rules"] = [m.rule.name for m in acted]
            notice["detail"] = self._detail(stage, matched)
        return notice

    def _detail(self, stage: Stage, matched: list[Matched]) -> dict[str, Any]:
        """Return every answer and match of `stage`, for a developer view in the client.

        Values are shown as the record stores them (`content.stored_value`). Entity
        names become "[found]", so the client never sees what a mask removed.
        """
        from django_ai_sdk.oversight.content import stored_value

        policy = self.resolved
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
                    "label": o.lens.display_label,
                    "kind": o.lens.answer.kind,
                    "value": stored_value(o.lens, o.answer),
                    "confidence": o.answer.confidence if o.answer.ok else None,
                    "answerer": o.answer.answerer,
                    "model": o.answer.model,
                    "error": o.answer.error,
                    "latency_ms": o.answer.latency_ms,
                }
                for o in self.answered
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
        """Return tool-stage matches to include in the finish notice."""
        if stage is not Stage.FINISH:
            return []
        return [m for m in self.matched if m.stage is Stage.TOOL and m not in matched]

    def notifications(self, matched: list[Matched]) -> list[tuple[dict[str, Any], list[str]]]:
        """Return (event, notifier names) for each enforced Notify action in `matched`."""
        from django_ai_sdk.oversight.notifiers import match_events

        return match_events(
            self.id,
            matched,
            agent=self.context.agent_name,
            channel=self.channel,
            outcome=str(self.outcome),
        )


def _seen(match: Matched) -> bool:
    """True when the person may learn that `match` acted, by a visible action or its notice."""
    told = bool(match.rule.notice) and not match.failed
    return told or any(a.kind in USER_FACING for a in match.acted)


def acted_on(
    matched: Iterable[Matched], restrict: list[str] | None = None
) -> tuple[str | None, list[str] | None]:
    """Return the first Block's refusal, and `restrict` with every RestrictTools' tools added."""
    block = None
    for match in matched:
        for action in match.acted:
            if isinstance(action, Block) and block is None:
                block = refusal(match, action)
            elif isinstance(action, RestrictTools):
                restrict = merge_restrict(restrict, action.names)
    return block, restrict


def refusal(match: Matched, block: Block) -> str:
    """Return the refusal message. A match caused only by a failure gets the generic one."""
    if match.failed:
        return str(Block.DEFAULT_MESSAGE)
    return str(match.rule.message or block.message)


def restrict_refusal(match: Matched) -> str:
    """Return the refusal for a tool `match` removed. Like `refusal`, a failure gets the generic one."""
    if match.failed:
        return str(TOOL_UNAVAILABLE)
    return str(match.rule.message or TOOL_UNAVAILABLE)
