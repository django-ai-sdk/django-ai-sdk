"""Compute model call costs from AI_SDK_MODEL_PRICES.

Prices are per million tokens, in any currency::

    AI_SDK_MODEL_PRICES = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "claude-": {"input": 3.00, "output": 15.00},   # every claude-* model
    }

A key matches a model exactly or as the longest prefix. Without a match the
cost is None, never a guess.
"""

from __future__ import annotations

from typing import Any

from django_ai_sdk.utils import resolve_setting


def model_price(model: str | None) -> dict[str, float] | None:
    """Return the price for `model` by exact key, else by longest prefix."""
    if not model:
        return None
    prices: dict[str, dict[str, float]] = resolve_setting("AI_SDK_MODEL_PRICES", {}) or {}
    if model in prices:
        return prices[model]
    # Strip provider prefixes, so "openai:gpt-4.1" matches "gpt-4.1".
    bare = model.split(":", 1)[-1].split("/", 1)[-1]
    if bare in prices:
        return prices[bare]
    matches = [key for key in prices if model.startswith(key) or bare.startswith(key)]
    return prices[max(matches, key=len)] if matches else None


def cost_for(
    model: str | None, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """Return the cost of one call, or None without a price or usage."""
    price = model_price(model)
    if price is None or (prompt_tokens is None and completion_tokens is None):
        return None
    total = (prompt_tokens or 0) * float(price.get("input", 0)) + (completion_tokens or 0) * float(
        price.get("output", 0)
    )
    return round(total / 1_000_000, 8)


def token_counts(usage: Any) -> tuple[int | None, int | None]:
    """Read (prompt, completion) tokens from a usage dict, for any vendor.

    The Responses API names them input/output tokens, Chat Completions prompt/completion.
    """
    if not isinstance(usage, dict):
        return None, None
    # A key set to None (some vendors send both names) falls back to the other.
    prompt = _first(usage.get("prompt_tokens"), usage.get("input_tokens"))
    completion = _first(usage.get("completion_tokens"), usage.get("output_tokens"))
    return (
        int(prompt) if isinstance(prompt, int | float) else None,
        int(completion) if isinstance(completion, int | float) else None,
    )


def _first(value: Any, other: Any) -> Any:
    return other if value is None else value


__all__ = ["cost_for", "model_price", "token_counts"]
