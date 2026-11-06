"""Workflow definitions, autodiscovered on startup.

Each is a named `WorkflowDefinition` passed to `register_workflow()`, which is what makes it
runnable by name. A step's result is filed under its own `name`, so a later step
requires it by that name and a listener reads it with `ctx.step(...)`.
"""

from __future__ import annotations

from django_ai_sdk.workflows.registry import register_workflow
from django_ai_sdk.workflows.schemas import (
    FieldDefinition,
    ListenerDefinition,
    StepDefinition,
    WorkflowDefinition,
)

from .pirate_basic import PirateBasicAgent

PIRATE = PirateBasicAgent().agent_id

# Open-Meteo needs no API key, so this runs with no credentials configured.
# The agent id comes from the class, so a typo is an ImportError here.
register_workflow(
    WorkflowDefinition(
        name="harbour-report",
        # A definition cannot replace the agent's system prompt, so the brief is the
        # step's user turn: a declared input with a default, sent as its history.
        input_fields={
            "brief": FieldDefinition(
                required=False,
                default=(
                    "Call the weather tool for Rotterdam and report the conditions as a "
                    "ship's log entry. Three sentences. Do not invent a forecast if the "
                    "tool returns an error — say so instead."
                ),
            )
        },
        steps=[StepDefinition(name="forecast", agent_id=PIRATE, history=["brief"])],
        # A workflow-level listener: it sees the run, and posts what `forecast` produced.
        listeners=[
            ListenerDefinition(
                type="thread_message", config={"agent_id": PIRATE, "step": "forecast"}
            )
        ],
    )
)

# Two steps, the second returning typed fields. No listener: the caller reads the run.
register_workflow(
    WorkflowDefinition(
        name="sailing-verdict",
        input_fields={
            "brief": FieldDefinition(
                required=False,
                default="Call the weather tool for Rotterdam and describe the conditions.",
            )
        },
        steps=[
            StepDefinition(name="forecast", agent_id=PIRATE, history=["brief"]),
            StepDefinition(
                name="verdict",
                agent_id=PIRATE,
                requires=["forecast"],
                output_fields={
                    "sailing": FieldDefinition(
                        type="str", description="good | risky | stay ashore"
                    ),
                    "windspeed_kmh": FieldDefinition(type="float"),
                },
            ),
        ],
    )
)


# Two host-supplied step types (see steps.py and AI_SDK_WORKFLOWS["STEPS"]): plain Python
# gathers the thread, then an agent step condenses it and retitles the thread.
register_workflow(
    WorkflowDefinition(
        name="thread-digest",
        # Declared, so a caller who forgets `thread` hears about it before the run
        # is queued rather than three steps in.
        input_fields={"thread": FieldDefinition(type="str", description="Thread id to digest")},
        steps=[
            StepDefinition(type="gather_thread", name="transcript"),
            StepDefinition(
                type="thread_digest",
                name="digest",
                requires=["transcript"],
                # A step-level listener: fires for this step alone.
                listeners=[ListenerDefinition(type="console_log")],
            ),
        ],
    )
)
