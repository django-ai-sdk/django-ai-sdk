"""Request and response schemas for the switch routes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from django_ai_sdk.oversight.services.access import Reason
from django_ai_sdk.oversight.types import Mode


class RuleModeIn(BaseModel):
    """Set a rule's mode in a policy. None hands control back to code."""

    mode: Mode | None
    # Required when the switch lowers the rule's mode.
    reason: Reason = ""


class AgentPolicyIn(BaseModel):
    """Set an agent's policy. "" turns oversight off, None hands control back to code."""

    policy: str | None = Field(..., max_length=100)
    # Required when the switch turns the agent's oversight off or moves it to another policy.
    reason: Reason = ""


class SwitchOut(BaseModel):
    """A rule or agent after a switch: the code value, the switched value and the effective one."""

    kind: str
    # The rule's policy for rule_mode, "" for an agent.
    policy: str = ""
    # The rule's name, or the agent's id.
    target: str
    # The value declared in code: the rule's mode or the agent class's policy.
    code: str
    # The switched value, or None when code decides.
    switched: str | None = None
    # What turns use once the switch reaches them (within RUNTIME_CACHE_SECONDS).
    effective: str
    reason: str = ""
    changed_by: str = ""
    changed_at: datetime | None = None
    # Why the switch does not apply: its rule, agent or policy is gone, or will not load.
    problem: str = ""


__all__ = ["AgentPolicyIn", "RuleModeIn", "SwitchOut"]
