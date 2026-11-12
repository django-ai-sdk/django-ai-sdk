"""A code policy to extend, operators who may author, and definitions to save."""

from __future__ import annotations

from typing import Any

from django_ai_sdk.oversight import Block, L, PatternLens, Policy, Rule

AUTHOR = "django_ai_sdk_oversight_authoring.change_authoredpolicy"
VIEW = "django_ai_sdk_oversight_authoring.view_authoredpolicy"


class Base(Policy):
    name = "au-base"
    lenses = [PatternLens(name="mentions_x", keywords=["xylophone"])]
    rules = [
        Rule("block_x", when=L("mentions_x").is_true, then=Block("No x."), mode="enforce")
    ]


AUTHORING_POLICIES = [Base]


def refunds(**changes: Any) -> dict[str, Any]:
    """A definition that blocks refund requests on top of au-base."""
    definition: dict[str, Any] = {
        "label": "Refunds",
        "extends": ["au-base"],
        "lenses": [
            {"kind": "pattern", "name": "refund", "stages": ["gate"], "keywords": ["refund"]}
        ],
        "rules": [
            {
                "name": "no_refunds",
                "when": {"lens": "refund", "op": "is_true"},
                "then": [{"kind": "block", "message": "Refunds go through support."}],
                "mode": "enforce",
            }
        ],
    }
    definition.update(changes)
    return definition
