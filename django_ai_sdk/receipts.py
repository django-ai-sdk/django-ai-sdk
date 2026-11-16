"""Tool receipts: what of a tool's result survives when oversight withholds the answer.

A withheld answer drops every tool call's arguments and result, which could reveal
it. A tool that starts something durable (a workflow run, a payout claim) declares
the keys that identify it, so the person can still find what was started:

    tool = with_receipt(start_workflow_run, "run_id", "workflow_id", "workflow", "status")

Only those keys are kept, and only short ids, slugs and enums or whole numbers.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from haystack.tools.utils import flatten_tools_or_toolsets

if TYPE_CHECKING:
    from collections.abc import Iterable

    from haystack.tools import Tool

# The attribute a tool declares its receipt keys in.
RECEIPT_KEYS = "receipt_keys"

# What a kept value may look like: an id, a slug or a short enum.
SAFE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def with_receipt(tool: Tool, *keys: str) -> Tool:
    """Declare the keys of `tool`'s result kept when the answer is withheld. Returns `tool`."""
    setattr(tool, RECEIPT_KEYS, tuple(keys))
    return tool


def receipt_keys(tools: Any) -> dict[str, tuple[str, ...]]:
    """The receipt keys of each tool in `tools` (tools and toolsets) that declares them, by name."""
    return {
        tool.name: tuple(keys)
        for tool in flatten_tools_or_toolsets(tools or [])
        if (keys := getattr(tool, RECEIPT_KEYS, None))
    }


def receipt(output: Any, keys: Iterable[str] | None) -> dict[str, Any] | None:
    """The receipt of a tool `output` (stored or streamed), marked "withheld"; None for none.

    A failed call has none, and a value that is not a short id-like string or a
    whole number is left out.
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
        return True
    return isinstance(value, str) and SAFE_VALUE.fullmatch(value) is not None
