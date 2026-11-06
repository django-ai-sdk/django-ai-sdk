"""Request and response schemas for the authoring routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from django_ai_sdk.oversight.services.access import Reason


class PolicyIn(BaseModel):
    """A policy definition plus the reason for saving it."""

    label: str = Field("", max_length=200)
    description: str = Field("", max_length=5_000)
    extends: list[str] = Field(default_factory=list, max_length=20)
    # e.g. {"kind": "judge", "name", "question", "judges": ["judge"], ...}
    lenses: list[dict[str, Any]] = Field(default_factory=list)
    # e.g. {"name", "when": {"lens": "x", "op": "is_true"}, "then": [{"kind": "block"}], ...}
    rules: list[dict[str, Any]] = Field(default_factory=list)
    # capture, stream_mode, fail_mode and retention. A missing key is inherited.
    options: dict[str, Any] = Field(default_factory=dict)
    reason: Reason

    def definition(self) -> dict[str, Any]:
        return self.model_dump(exclude={"reason"})


class WithdrawIn(BaseModel):
    reason: Reason


class TryIn(BaseModel):
    """One exchange for a dry run of a definition. Nothing is recorded."""

    definition: PolicyIn | None = None
    input: str = Field(..., max_length=100_000)
    output: str = Field("", max_length=100_000)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class VersionOut(BaseModel):
    version: int
    withdrawn: bool
    reason: str
    changed_by: str
    changed_at: datetime


class PolicyOut(BaseModel):
    """The newest version of an authored policy, and whether this process has it loaded."""

    name: str
    version: int
    label: str
    description: str
    extends: list[str]
    lenses: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    options: dict[str, Any]
    withdrawn: bool
    # Loaded in this process. Other processes catch up within RUNTIME_CACHE_SECONDS.
    loaded: bool
    # Why the newest version failed to load, or "".
    error: str = ""
    reason: str = ""
    changed_by: str = ""
    changed_at: datetime | None = None
    versions: list[VersionOut] = Field(default_factory=list)


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
    # Answer types a lens may give: "flag", "choice", "score", "entities".
    answers: list[str]
    actions: list[str]
    modes: list[str]
    operators: list[str]
    stages: list[str]
    subjects: list[str]
    # Names of the judge aliases the deployment set up.
    aliases: list[str]
    # What a lens may configure inline ("keywords" and presets), with the arguments each takes.
    inline: dict[str, list[str]]
    # Names of the validators a check lens may use.
    validators: list[str]
    # Policies a definition may extend: code policies, packs and authored ones.
    policies: list[dict[str, str]]
    must_extend: list[str]


__all__ = [
    "CatalogOut",
    "PolicyIn",
    "PolicyOut",
    "StageOut",
    "TryIn",
    "TryOut",
    "VersionOut",
    "WithdrawIn",
]
