"""Workflow declarations for the demo application, autodiscovered on startup.

A workflow declared here is reachable from the chat API, the workflow endpoints, and the
automations in automations.py that name it.
"""

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
# The agent id comes from the class, so a typo is an ImportError here.
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

# Two steps, the second returning typed fields. No action: the automation running this
# one is app-level, so its result is recorded on the AutomationRun.
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
