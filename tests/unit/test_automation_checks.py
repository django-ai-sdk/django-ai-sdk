"""System checks: the failure modes logging alone cannot cover.

Every one is a Warning; booting is never blocked by a background job's configuration.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from django_ai_sdk.automations import Automation
from django_ai_sdk.automations.checks import check_automations
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.workflows import FieldDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.registry import register as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows

# A registry key, the shape a real reference has.
AGENT_ID = "663dd70f-65d2-58c4-9edc-037c9904d562"

WORKFLOW = "some-workflow"


@pytest.fixture(autouse=True)
def _clean_registries(settings):
    # The demo configures its own automations, which would read as orphaned config.
    settings.AI_SDK_AUTOMATIONS = {}
    reset_registry()
    reset_workflows()
    yield
    reset_registry()
    reset_workflows()


def ids(issues) -> set[str]:
    return {issue.id for issue in issues}


def register_workflow(name=WORKFLOW, agent_id=AGENT_ID, *, takes_input=True):
    """A workflow an automation can name, declaring the turn the automation writes."""
    return register_wf(
        WorkflowDefinition(
            name=name,
            input_fields=({"messages": FieldDefinition(type="list")} if takes_input else {}),
            steps=[
                StepDefinition(
                    name="result", agent_id=agent_id, history=["messages"] if takes_input else []
                )
            ],
        )
    )


def declare(**attrs):
    defaults = {"name": "example", "cron": "0 9 * * *", "workflow": WORKFLOW}
    register(type("Example", (Automation,), {**defaults, **attrs}))


class TestChecks:
    def test_nothing_declared_reports_nothing(self):
        assert check_automations() == []

    def test_a_fully_wired_automation_reports_nothing(self):
        register_workflow()
        declare()
        assert check_automations() == []

    def test_a_refused_declaration_is_reported(self):
        @register
        class Broken(Automation):
            name = "broken"
            cron = "not a cron"
            workflow = "some-workflow"

        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W006"]
        assert "not a valid 5-field cron expression" in issue.msg

    def test_a_refused_declaration_is_not_also_an_orphaned_setting(self, settings):
        settings.AI_SDK_AUTOMATIONS = {"broken": {"ENABLED": True}}

        @register
        class Broken(Automation):
            name = "broken"
            workflow = ""

        assert "ai_sdk.automations.W005" not in ids(check_automations())

    def test_an_unusable_schedule_is_reported(self):
        register_workflow()
        declare()
        # Registration validated the schedule; settings can still break it afterwards,
        # which is the case the check exists for.
        with patch.object(Automation, "get_schedule", side_effect=RuntimeError("nope")):
            assert "ai_sdk.automations.W001" in ids(check_automations())

    def test_an_unknown_timezone_is_reported(self):
        register_workflow()
        declare(timezone="Mars/Olympus_Mons")
        assert "ai_sdk.automations.W002" in ids(check_automations())

    def test_a_known_timezone_is_not(self):
        register_workflow()
        declare(timezone="Europe/Amsterdam")
        assert "ai_sdk.automations.W002" not in ids(check_automations())

    def test_settings_configuring_a_name_nothing_declares(self, settings):
        settings.AI_SDK_AUTOMATIONS = {"typoed-name": {"ENABLED": False}}
        register_workflow()
        declare()
        assert "ai_sdk.automations.W005" in ids(check_automations())

    def test_every_issue_is_a_warning_never_an_error(self):
        declare(requires=["nonexistent"])
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


class TestTheWorkflowCanReceiveTheTurn:
    """A schedule has nobody typing, so the automation writes the first message.

    The workflow has to declare an input to receive it — an input a definition does
    not declare is dropped before the first step, and the run then succeeds having
    asked the model nothing. That is the failure this check exists to catch, because
    nothing else would report it.
    """

    def test_a_workflow_declaring_the_input_passes(self):
        register_workflow()
        declare()
        assert "ai_sdk.automations.W007" not in ids(check_automations())

    def test_a_workflow_declaring_no_inputs_is_reported(self):
        register_workflow(takes_input=False)
        declare()
        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W007"]
        assert "will be dropped" in issue.msg

    def test_an_automation_pointing_at_the_wrong_input_is_reported(self):
        register_workflow()
        declare(input_name="prompt")
        assert "ai_sdk.automations.W007" in ids(check_automations())

    def test_an_input_of_the_wrong_type_is_reported(self):
        """Declared, but as an `int`: nothing the automation seeds can land there."""
        register_wf(
            WorkflowDefinition(
                name=WORKFLOW,
                input_fields={"messages": FieldDefinition(type="int")},
                steps=[StepDefinition(name="result", agent_id=AGENT_ID, history=["messages"])],
            )
        )
        declare()
        assert "ai_sdk.automations.W007" in ids(check_automations())

    def test_a_str_input_receives_the_turn_as_is(self):
        register_wf(
            WorkflowDefinition(
                name=WORKFLOW,
                input_fields={"messages": FieldDefinition(type="str")},
                steps=[StepDefinition(name="result", agent_id=AGENT_ID, history=["messages"])],
            )
        )
        declare()
        assert "ai_sdk.automations.W007" not in ids(check_automations())

    def test_an_input_no_step_reads_is_reported(self):
        """Declared and seedable, but no agent step wires it as history."""
        register_wf(
            WorkflowDefinition(
                name=WORKFLOW,
                input_fields={"messages": FieldDefinition(type="list")},
                steps=[StepDefinition(name="result", agent_id=AGENT_ID)],
            )
        )
        declare()
        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W007"]
        assert "history" in issue.hint

    def test_an_input_only_a_registered_step_reads_is_not_reported(self):
        """A python step may read the input itself, which no check can see through."""
        register_wf(
            WorkflowDefinition(
                name=WORKFLOW,
                input_fields={"messages": FieldDefinition(type="list")},
                steps=[StepDefinition(type="shout", name="result")],
            )
        )
        declare()
        assert "ai_sdk.automations.W007" not in ids(check_automations())

    def test_a_workflow_nothing_declares_says_only_that(self):
        """W003 already covers it; a second warning about its inputs adds nothing."""
        declare(workflow="no-such-workflow")
        assert "ai_sdk.automations.W007" not in ids(check_automations())


class TestRequires:
    def test_requiring_an_unregistered_integration(self):
        register_workflow()
        declare(requires=["nonexistent"])
        [issue] = [i for i in check_automations() if i.id == "ai_sdk.automations.W004"]
        assert "nonexistent" in issue.msg

    def test_a_registered_integration_passes(self):
        from django_ai_sdk.integrations import registry as integrations_registry

        class FakeIntegration:
            name = "notion"

        register_workflow()
        declare(requires=["notion"])
        with patch.dict(integrations_registry._registry, {"notion": FakeIntegration()}):
            assert "ai_sdk.automations.W004" not in ids(check_automations())
