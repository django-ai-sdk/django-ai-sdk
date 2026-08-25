"""Workflow declarations for the demo application, autodiscovered on startup."""

from __future__ import annotations

from django_ai_sdk.workflows import (
    WorkflowAction,
    WorkflowDefinition,
    WorkflowStep,
    register,
)

from .pirate_basic import PirateBasicAgent

PIRATE = PirateBasicAgent().agent_id

# Open-Meteo needs no API key, so this runs with no credentials configured.
register(
    WorkflowDefinition(
        name="harbour-report",
        steps=[
            WorkflowStep(
                name="forecast",
                agent_id=PIRATE,
                output_key="report",
                system_prompt_override=(
                    "Call the weather tool for Rotterdam and report the conditions as a "
                    "ship's log entry. Three sentences. Do not invent a forecast if the "
                    "tool returns an error — say so instead."
                ),
            )
        ],
        actions=[WorkflowAction(type="thread_message", input_key="report")],
    )
)

# No action, so the result is read back from the WorkflowRun.
register(
    WorkflowDefinition(
        name="sailing-verdict",
        steps=[
            WorkflowStep(
                name="forecast",
                agent_id=PIRATE,
                output_key="report",
                system_prompt_override=(
                    "Call the weather tool for Rotterdam and describe the conditions."
                ),
            ),
            WorkflowStep(
                name="judge",
                agent_id=PIRATE,
                input_key="report",
                output_key="verdict",
                output_fields={
                    "sailing": {"type": "str", "description": "good | risky | stay ashore"},
                    "windspeed_kmh": {"type": "float"},
                },
            ),
        ],
    )
)
