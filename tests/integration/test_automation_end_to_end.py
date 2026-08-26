"""The whole path: a due automation becomes a message in someone's chat.

Only the model call is faked; everything between the schedule and the thread is real.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.automations import Audience, Automation
from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.automations.runner import tick
from django_ai_sdk.automations.tasks import run_automation
from django_ai_sdk.conversation.models import Message, Thread
from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.workflows import WorkflowAction, WorkflowDefinition, WorkflowStep
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.registry import register as register_workflow
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows

NOW = datetime(2026, 8, 17, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    yield
    reset_registry()
    reset_workflows()


def declare_workflow(name="ships-log", steps=1):
    """The workflow the automations below reference. One step, or two."""
    if steps == 1:
        definition = WorkflowDefinition(
            name=name,
            steps=[WorkflowStep(name="write", agent_id="log-keeper", output_key="entry")],
            actions=[WorkflowAction(type="thread_message", input_key="entry")],
        )
    else:
        definition = WorkflowDefinition(
            name=name,
            steps=[
                WorkflowStep(name="summarise", agent_id="log-keeper", output_key="summary"),
                WorkflowStep(
                    name="classify",
                    agent_id="log-keeper",
                    input_key="summary",
                    output_key="verdict",
                ),
            ],
            actions=[WorkflowAction(type="thread_message", input_key="verdict")],
        )
    return register_workflow(definition)


@pytest.fixture
def agent_says():
    """Patch agent resolution so the workflow runs without a model call.

    The storage adapter stays real: a mocked one hands back a mock id and nothing
    persists.
    """

    def _configure(text="Two ships sighted, no losses."):
        agent = MagicMock()
        agent.run = AsyncMock(return_value=text)
        agent.id = "log-keeper"
        agent.name = "Log keeper"
        agent.model = "gpt-4o-mini"
        agent.permissions = [AllowAll]
        agent.storage_adapter = DbStorageAdapter
        # Off so tests can assert on the one agent.run the workflow itself makes.
        agent.title_generation = False
        return patch(
            "django_ai_sdk.workflows.executor.AgentService.get",
            AsyncMock(return_value=agent),
        )

    return _configure


def declare(name="ships-log", **attrs) -> type[Automation]:
    """An automation naming the workflow declared above. Registers the workflow too."""
    attrs.setdefault("workflow", "ships-log")
    if attrs["workflow"] == "ships-log":
        declare_workflow()
    defaults = {
        "name": name,
        "cron": "0 9 * * *",
        "input": "Write the log entry for everything since {last_run_at}.",
    }
    cls = type("Example", (Automation,), {**defaults, **attrs})
    register(cls)
    return cls


async def make_users(*names):
    from tests.factories.db import UserFactory

    return [await UserFactory.acreate(email=f"{name}@example.com") for name in names]


async def subscribe(*users, name="ships-log"):
    """Opt every given user into the named automation."""
    from django_ai_sdk.automations.models import AutomationSubscription

    for user in users:
        await AutomationSubscription.objects.acreate(name=name, user=user, enabled=True)


async def run_everything_due(now=NOW):
    """One tick, then execute every run it produced — what cron plus a worker do."""
    results = await tick(now=now)
    for result in results:
        for run in result.runs:
            if run.status == AutomationRun.Status.PENDING:
                await run_automation(str(run.id))
    return results


@pytest.mark.django_db(transaction=True)
class TestScheduleToChatMessage:
    async def test_a_due_automation_delivers_a_message_to_its_user(self, agent_says):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)

        await tick(now=NOW)  # bootstrap only; nothing is due on first sight
        await AutomationState.objects.aupdate(next_run_at=NOW)

        with agent_says("Two ships sighted, no losses."):
            await run_everything_due()

        # What the user sees.
        thread = await Thread.objects.filter(user=user).afirst()
        assert thread is not None
        message = await Message.objects.filter(thread=thread).afirst()
        assert "Two ships sighted" in str(message.result)

    async def test_the_run_records_success_and_links_its_workflow(self, agent_says):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        with agent_says():
            await run_everything_due()

        run = await AutomationRun.objects.aget(name="ships-log")
        assert run.status == AutomationRun.Status.SUCCEEDED
        assert run.started_at is not None
        assert run.finished_at is not None
        # The AutomationRun must be able to answer "what did it actually do?", which
        # means reaching the per-step detail.
        assert run.workflow_run_id is not None
        assert await WorkflowRun.objects.filter(id=run.workflow_run_id).aexists()

    async def test_the_lease_is_released_and_success_recorded(self, agent_says):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        with agent_says():
            await run_everything_due()

        state = await AutomationState.objects.aget(name="ships-log")
        assert state.locked_until is None
        # last_success_at is what the next run's {last_run_at} reads.
        assert state.last_success_at is not None
        assert state.next_run_at > NOW

    async def test_the_input_reaches_the_agent_with_placeholders_filled(self, agent_says):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        patcher = agent_says()
        with patcher as mocked:
            await run_everything_due()

        agent = await mocked()
        assert agent.run.await_count == 1
        # First run, so there is no previous success to be incremental from.
        rendered = str(agent.run.await_args)
        assert "{last_run_at}" not in rendered


@pytest.mark.django_db(transaction=True)
class TestFanOut:
    async def test_every_user_gets_their_own_thread(self, agent_says):
        users = await make_users("ann", "bo", "cy")
        await subscribe(*users)

        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        with agent_says():
            await run_everything_due()

        assert await AutomationRun.objects.acount() == 3
        for user in users:
            assert await Thread.objects.filter(user=user).acount() == 1

        # One tick is one logical dispatch, however many rows it produced.
        ids = {r.dispatch_id async for r in AutomationRun.objects.all()}
        assert len(ids) == 1


@pytest.mark.django_db(transaction=True)
class TestFailurePath:
    async def test_a_broken_agent_fails_the_run_and_frees_the_lease(self):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        exploding = AsyncMock(side_effect=ValueError("Agent 'log-keeper' not found"))
        with patch("django_ai_sdk.workflows.executor.AgentService.get", exploding):
            results = await tick(now=NOW)
            [run] = results[0].runs
            with pytest.raises(Exception):
                # Re-raised so django-tasks marks its own result failed too.
                await run_automation(str(run.id))

        run = await AutomationRun.objects.aget(id=run.id)
        assert run.status == AutomationRun.Status.FAILED
        assert "log-keeper" in run.error

        # A failed run must not wedge the automation until its lease expires...
        state = await AutomationState.objects.aget(name="ships-log")
        assert state.locked_until is None
        # ...nor claim a window it never processed.
        assert state.last_success_at is None


    async def test_a_degraded_integration_skips_rather_than_fails(self, agent_says):
        from django_ai_sdk.integrations.base import IntegrationStatus

        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED, requires=["notion"])
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        sick = AsyncMock()
        sick.get_status = AsyncMock(return_value=IntegrationStatus.DEGRADED)
        with patch(
            "django_ai_sdk.integrations.registry.get_integrations",
            AsyncMock(return_value={"notion": sick}),
        ):
            await run_everything_due()

        run = await AutomationRun.objects.aget(name="ships-log")
        # An outage upstream is not a broken automation.
        assert run.status == AutomationRun.Status.SKIPPED
        assert "notion" in run.skip_reason
        assert await Thread.objects.acount() == 0


@pytest.mark.django_db(transaction=True)
class TestMultiStepWorkflow:
    async def test_a_two_step_workflow_runs_both_steps_and_delivers(self, agent_says):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare_workflow(name="weekly-review", steps=2)
        declare(audience=Audience.SUBSCRIBED, workflow="weekly-review")
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        patcher = agent_says("all clear")
        with patcher as mocked:
            await run_everything_due()

        agent = await mocked()
        assert agent.run.await_count == 2

        run = await AutomationRun.objects.aget(name="ships-log")
        assert run.status == AutomationRun.Status.SUCCEEDED
        assert set(run.output) == {"summary", "verdict"}
        assert await Thread.objects.acount() == 1


@pytest.mark.django_db(transaction=True)
class TestThreadTitle:
    async def test_a_delivered_thread_gets_a_generated_title_not_the_raw_workflow_name(self):
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED)
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        agent = MagicMock()
        # First call is the workflow step, second is title generation.
        agent.run = AsyncMock(
            side_effect=["Two ships sighted, no losses.", "Ship sighting report"]
        )
        agent.id = "log-keeper"
        agent.name = "Log keeper"
        agent.model = "gpt-4o-mini"
        agent.permissions = [AllowAll]
        agent.storage_adapter = DbStorageAdapter
        agent.title_generation = True
        agent.get_title_generation_prompt = MagicMock(return_value="Give this a short title.")

        with patch(
            "django_ai_sdk.agents.services.AgentService.get",
            AsyncMock(return_value=agent),
        ):
            await run_everything_due()

        thread = await Thread.objects.filter(user=user).afirst()
        assert thread.title == "Ship sighting report"


@pytest.mark.django_db(transaction=True)
class TestMissingWorkflow:
    async def test_a_run_naming_an_unregistered_workflow_is_skipped(self):
        # The automation is not broken, its dependency is absent — the same
        # distinction a degraded integration gets.
        [user] = await make_users("bosun")
        await subscribe(user)
        declare(audience=Audience.SUBSCRIBED, workflow="vanished")
        await tick(now=NOW)
        await AutomationState.objects.aupdate(next_run_at=NOW)

        await run_everything_due()

        run = await AutomationRun.objects.aget(name="ships-log")
        assert run.status == AutomationRun.Status.SKIPPED
        assert "vanished" in run.skip_reason
        assert await Thread.objects.acount() == 0
