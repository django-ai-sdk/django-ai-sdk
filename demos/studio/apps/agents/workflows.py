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

register_workflow(
    WorkflowDefinition(
        name="harbour-report",
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
        listeners=[
            ListenerDefinition(
                type="thread_message", config={"agent_id": PIRATE, "step": "forecast"}
            )
        ],
    )
)

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


register_workflow(
    WorkflowDefinition(
        name="thread-digest",
        input_fields={"thread": FieldDefinition(type="str", description="Thread id to digest")},
        steps=[
            StepDefinition(type="gather_thread", name="transcript"),
            StepDefinition(
                type="thread_digest",
                name="digest",
                requires=["transcript"],
                listeners=[ListenerDefinition(type="console_log")],
            ),
        ],
    )
)
