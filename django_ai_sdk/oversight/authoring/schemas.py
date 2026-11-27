from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from django_ai_sdk.oversight.types import Mode


class PolicyIn(BaseModel):
    """A policy definition. Saving it replaces the policy's definition."""

    label: str = Field("", max_length=200)
    description: str = Field("", max_length=5_000)
    extends: list[str] = Field(default_factory=list)
    lenses: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class ActiveIn(BaseModel):
    """Turn the policy's own lenses and rules on or off. Its parents always run."""

    active: bool


class RuleModeIn(BaseModel):
    """Set the mode of one of the policy's own rules."""

    mode: Mode


class LensEnabledIn(BaseModel):
    """Turn one of the policy's own lenses on or off."""

    enabled: bool


class TryIn(BaseModel):
    """One exchange for a dry run of a definition. Nothing is recorded."""

    definition: PolicyIn | None = None
    input: str = Field(..., max_length=100_000)
    output: str = Field("", max_length=100_000)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class PolicyOut(BaseModel):
    """An authored policy, and whether this process has it loaded."""

    name: str
    label: str
    description: str
    extends: list[str]
    lenses: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    options: dict[str, Any]
    active: bool
    loaded: bool
    error: str = ""
    updated_by: str = ""
    updated_at: datetime | None = None


class StageOut(BaseModel):
    stage: str
    blocked: bool = False
    message: str = ""
    text: str | None = None
    tool: str = ""
    answers: dict[str, Any] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)


class TryOut(BaseModel):
    outcome: str
    report_outcome: str
    ended_at: str
    stages: list[StageOut]


class CatalogOut(BaseModel):
    """The building blocks a definition may use, for an editor UI."""

    lens_kinds: list[str]
    answers: list[str]
    actions: list[str]
    modes: list[str]
    operators: list[str]
    stages: list[str]
    subjects: list[str]
    aliases: list[str]
    inline: dict[str, list[str]]
    validators: list[str]
    policies: list[dict[str, str]]
    must_extend: list[str]


__all__ = [
    "ActiveIn",
    "CatalogOut",
    "LensEnabledIn",
    "PolicyIn",
    "PolicyOut",
    "RuleModeIn",
    "StageOut",
    "TryIn",
    "TryOut",
]
