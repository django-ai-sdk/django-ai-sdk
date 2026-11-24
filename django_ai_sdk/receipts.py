"""Tool receipts: what of a tool's result survives when oversight withholds the answer.

A withheld answer drops every tool call's arguments and result, which could reveal
it. A tool that starts something durable (a workflow run, a payout claim) declares
the keys that identify it, so the person can still find what was started:

    tool = with_receipt(start_workflow_run, "run_id", "workflow_id", "workflow", "status")

The keys are trusted as the tool's author declared them; their values are still
screened: only short ids, slugs and enums or whole numbers are kept, and none that
looks like personal data or a secret.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from haystack.tools.utils import flatten_tools_or_toolsets

from django_ai_sdk.oversight.patterns import finds_personal_or_secret

if TYPE_CHECKING:
    from collections.abc import Iterable

    from haystack.tools import Tool

# What a kept value may look like: an id, a slug or a short enum.
SAFE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def with_receipt(tool: Tool, *keys: str) -> Tool:
    """Declare the keys of `tool`'s result kept when the answer is withheld. Returns `tool`."""
    tool.receipt_keys = tuple(keys)  # ty: ignore[missing-attribute]
    return tool


def receipt_keys(tools: Any) -> dict[str, tuple[str, ...]]:
    """The receipt keys of each tool in `tools` (tools and toolsets) that declares them, by name.

    Includes the tools of subagents they delegate to, at any depth; tools that
    share a name keep the keys of each.
    """
    found: dict[str, tuple[str, ...]] = {}
    for tool in flatten_tools_or_toolsets(tools or []):
        # Haystack keeps a ComponentTool's component private; a subagent reports its own.
        delegate = getattr(getattr(tool, "_component", None), "receipt_keys", None)
        nested = delegate() if callable(delegate) else {}
        for name, keys in [*nested.items(), (tool.name, getattr(tool, "receipt_keys", ()))]:
            if keys:
                found[name] = tuple(dict.fromkeys([*found.get(name, ()), *keys]))
    return found


def receipt(output: Any, keys: Iterable[str] | None) -> dict[str, Any] | None:
    """The receipt of a tool `output` (stored or streamed), marked "withheld"; None for none.

    A failed call has none, and a value that is not a short id-like string or a
    whole number, or that looks like personal data or a secret, is left out.
    """
    if not keys:
        return None
    if isinstance(output, dict) and "origin" in output:
        if output.get("error"):
            return None
        output = output.get("result")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    if not isinstance(output, dict):
        return None
    kept = {key: output[key] for key in keys if _safe(output.get(key))}
    return {**kept, "withheld": True} if kept else None


def _safe(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        value = str(value)
    elif not (isinstance(value, str) and SAFE_VALUE.fullmatch(value)):
        return False
    return not finds_personal_or_secret(value)
