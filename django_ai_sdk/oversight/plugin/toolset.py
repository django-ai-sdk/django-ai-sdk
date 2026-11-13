"""The tools a run may offer the model: the agent's own, less what the turn took away."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack.tools import Tool, Toolset

from django_ai_sdk.oversight.guard import current_guard

if TYPE_CHECKING:
    from collections.abc import Iterator


class GuardedToolset(Toolset):
    """The agent's tools, less the ones the current turn took away.

    Haystack flattens the tools again on every step, so a tool a rule takes away
    mid-run is gone from the model's next call. Warm-up, close and per-run copies
    are passed on to the wrapped Tools and Toolsets.
    """

    def __init__(self, items: list[Tool | Toolset]) -> None:
        # Not the dataclass init: the wrapped items are Toolsets too, not only Tools.
        self.tools = []
        self.items = items

    @classmethod
    def wrap(cls, tools: Any) -> Any:
        """Wrap `tools` once. No tools stay no tools."""
        if isinstance(tools, GuardedToolset) or not _has_items(tools):
            return tools
        return cls([tools] if isinstance(tools, Toolset) else list(tools))

    def _all(self) -> list[Tool]:
        return [
            tool for item in self.items for tool in (item if isinstance(item, Toolset) else [item])
        ]

    def __iter__(self) -> Iterator[Tool]:
        guard = current_guard()
        return iter([t for t in self._all() if guard is None or guard.allows_tool(t.name)])

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __contains__(self, item: str | Tool) -> bool:
        name = item if isinstance(item, str) else item.name
        return any(tool.name == name for tool in self)

    def get_selectable_tools(self) -> list[Tool]:
        return [
            tool
            for item in self.items
            for tool in (item.get_selectable_tools() if isinstance(item, Toolset) else [item])
        ]

    def spawn(self, selected_tool_names: set[str] | None = None) -> GuardedToolset:
        return GuardedToolset(
            [
                item.spawn(selected_tool_names) if isinstance(item, Toolset) else item
                for item in self.items
            ]
        )

    def warm_up(self) -> None:
        from haystack.tools.utils import warm_up_tools

        warm_up_tools(self.items)

    async def warm_up_async(self) -> None:
        from haystack.tools.utils import warm_up_tools_async

        await warm_up_tools_async(self.items)

    def close(self) -> None:
        from haystack.tools.utils import close_tools

        close_tools(self.items)

    async def close_async(self) -> None:
        from haystack.tools.utils import close_tools_async

        await close_tools_async(self.items)


def _has_items(tools: Any) -> bool:
    # Not truthiness: it would iterate a lazy Toolset and warm it up early.
    return tools is not None and (isinstance(tools, Toolset) or bool(list(tools)))


__all__ = ["GuardedToolset"]
