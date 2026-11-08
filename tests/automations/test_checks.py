"""System checks: the failure modes logging alone cannot cover.

Every one is a Warning; booting is never blocked by a background job's configuration.
"""

from __future__ import annotations

import pytest

from django_ai_sdk.automations import Automation
from django_ai_sdk.automations.checks import check_automations
from django_ai_sdk.automations.registry import register_automation, reset_registry
from django_ai_sdk.workflows.schemas import FieldDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.registry import register_workflow as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows

# A registry key, the shape a real reference has.
AGENT_ID = "663dd70f-65d2-58c4-9edc-037c9904d562"

WORKFLOW = "some-workflow"


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    yield
    reset_registry()
    reset_workflows()


def ids(issues) -> set[str]:
    return {issue.id for issue in issues}


def register_workflow(name=WORKFLOW, agent_id=AGENT_ID):
    """A workflow an automation can name, declaring the turn the automation writes."""
    return register_wf(
        WorkflowDefinition(
            name=name,
            input_fields={"prompt": FieldDefinition(type="str")},
            steps=[StepDefinition(name="result", agent_id=agent_id, history=["prompt"])],
        )
    )


def declare(**attrs):
    defaults = {"name": "example", "cron": "0 9 * * *", "workflow": WORKFLOW}
    register_automation(type("Example", (Automation,), {**defaults, **attrs}))


class TestChecks:
    def test_nothing_declared_reports_nothing(self):
        assert check_automations() == []

    def test_a_fully_wired_automation_reports_nothing(self):
        register_workflow()
        declare()
        assert check_automations() == []

    def test_a_refused_declaration_is_reported(self):
        @register_automation
        class Broken(Automation):
            name = "broken"
            cron = "not a cron"
            workflow = "some-workflow"

        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W006"]
        assert "not a valid 5-field cron expression" in issue.msg

    def test_every_issue_is_a_warning_never_an_error(self):
        declare(workflow="no-such-workflow")
        declare(name="broken", cron="nope")
        # Boot must not fail because a background job is misconfigured.
        assert all(issue.level < 40 for issue in check_automations())


class TestWorkflowReference:
    def test_a_workflow_nothing_declares_is_reported(self):
        declare(workflow="no-such-workflow")
        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W003"]
        assert "no-such-workflow" in issue.msg

    def test_a_declared_workflow_passes(self):
        register_workflow()
        declare()
        assert "ai_sdk.automations.W003" not in ids(check_automations())
