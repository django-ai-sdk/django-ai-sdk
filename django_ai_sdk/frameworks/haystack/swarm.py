"""Haystack binding for peer-to-peer agent handoffs (swarm pattern).

This module creates Haystack-compatible tools for agents to delegate tasks to
other agents. The core invocation logic (handoff.py) is framework-agnostic;
this module is the Haystack-specific integration.

To add support for other frameworks (LangChain, OpenAI native, etc), create
similar modules in their respective integration folders.
"""

from typing import Any

from django_ai_sdk.helpers.handoff import invoke_assistant_for_query
from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


def make_handoff_tool(agent_map: dict[str, type]) -> Any:
    """
    Create a Haystack Tool for peer-to-peer agent handoffs (swarm pattern).

    Any agent can use this tool to hand off to another agent, enabling
    direct agent-to-agent handoffs without a central orchestrator.

    Args:
        agent_map: Mapping of agent name → Assistant class (with _assistant_id)

    Returns:
        Haystack Tool for handing off to other agents
    """
    from haystack.tools import Tool

    agents_doc = "\n".join(f"- {name}" for name in agent_map.keys())

    def handoff_to_agent(agent: str, query: str) -> str:
        """Hand off a task to another agent."""
        entry = agent_map.get(agent.lower())
        if not entry:
            return f"Unknown agent '{agent}'. Available: {agents_doc}"

        cls = entry if isinstance(entry, type) else entry[0]
        result = invoke_assistant_for_query(
            cls._assistant_id,
            query=query,
            specialist=agent,
        )
        return result.answer or f"[{agent} error: {result.error}]"

    return Tool(
        name="handoff_to_agent",
        description=f"Hand off a task to another agent.\n\nAvailable agents:\n{agents_doc}",
        parameters={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": list(agent_map.keys()),
                    "description": "Which agent to hand off to.",
                },
                "query": {
                    "type": "string",
                    "description": "The task or question for the agent.",
                },
            },
            "required": ["agent", "query"],
        },
        function=handoff_to_agent,
    )
