"""Workflow definitions, autodiscovered on startup.

Each is a named `WorkflowDefinition` passed to `register()`, which is what makes it
runnable by name. A step's result is filed under its own `name`, so a later step
requires it by that name and a hook reads it with `ctx.step(...)`.
"""

from __future__ import annotations

from django_ai_sdk.workflows import (
    FieldDefinition,
    HookDefinition,
    StepDefinition,
    WorkflowDefinition,
    register,
)

from .pirate_basic import PirateBasicAgent

PIRATE = PirateBasicAgent().agent_id

# Open-Meteo needs no API key, so this runs with no credentials configured.
# The agent id comes from the class, so a typo is an ImportError here.
register(
    WorkflowDefinition(
        name="harbour-report",
        # The turn the run starts from. `daily-harbour-report` in automations.py
        # writes it on a schedule; an API caller supplies it directly.
        input_fields={"messages": FieldDefinition(type="messages")},
        steps=[
            StepDefinition(
                name="forecast",
                agent_id=PIRATE,
                system_prompt_override=(
                    "Call the weather tool for Rotterdam and report the conditions as a "
                    "ship's log entry. Three sentences. Do not invent a forecast if the "
                    "tool returns an error — say so instead."
                ),
            )
        ],
        # A workflow-level hook: it sees the run, and posts what `forecast` produced.
        hooks=[
            HookDefinition(type="thread_message", config={"agent_id": PIRATE, "step": "forecast"})
        ],
    )
)

# Two steps, the second returning typed fields. No hook: the caller reads the run.
register(
    WorkflowDefinition(
        name="sailing-verdict",
        input_fields={"messages": FieldDefinition(type="messages")},
        steps=[
            StepDefinition(
                name="forecast",
                agent_id=PIRATE,
                system_prompt_override=(
                    "Call the weather tool for Rotterdam and describe the conditions."
                ),
            ),
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


# Two host-supplied step types (see steps.py and AI_SDK_WORKFLOW_STEPS): plain Python
# gathers the thread, then an agent step condenses it and retitles the thread.
register(
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
                # A step-level hook: fires for this step alone.
                hooks=[HookDefinition(type="console_log")],
            ),
        ],
    )
)
