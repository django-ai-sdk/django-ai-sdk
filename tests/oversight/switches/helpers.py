"""Policies to switch, operators who may switch them, and the events a switch sends."""

from __future__ import annotations

from typing import Any

from django_ai_sdk.oversight import Alert, Block, L, PatternLens, Policy, Rule

SWITCH = "django_ai_sdk_oversight_switches.change_override"
VIEW = "django_ai_sdk_oversight_switches.view_override"


class SwitchParent(Policy):
    name = "sw-parent"
    lenses = [PatternLens(name="mentions_x", keywords=["xylophone"])]
    rules = [Rule("block_x", when=L("mentions_x").is_true, then=Block("No x."))]


class SwitchChild(Policy):
    name = "sw-child"
    extends = ["sw-parent"]  # noqa: RUF012
    rules = [Rule("note_x", when=L("mentions_x").is_true, then=Alert("info"), mode="enforce")]


class SwitchFloored(Policy):
    name = "sw-floored"
    lenses = [PatternLens(name="mentions_x", keywords=["xylophone"])]
    rules = [
        Rule(
            "always_block_x",
            when=L("mentions_x").is_true,
            then=Block("No x."),
            mode="enforce",
            min_mode="enforce",
        ),
        Rule(
            "watch_x",
            when=L("mentions_x").is_true,
            then=Alert("info"),
            mode="enforce",
            min_mode="report",
        ),
    ]


SWITCH_POLICIES = [SwitchParent, SwitchChild, SwitchFloored]


def events(sent: Any) -> list[dict[str, Any]]:
    """The events a patched `enqueue_notifications` was handed, in order."""
    return [event for call in sent.call_args_list for event, _ in call.args[0]]
