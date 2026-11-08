"""Workers racing for one run, each on its own connection. SQLite serialises every
writer, so these need PostgreSQL: set `DATABASE_URL` (see tests/settings.py)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from asgiref.sync import async_to_sync
from django.core.management import call_command
from django.utils import timezone

from django_ai_sdk.workflows.models import StepRun, WorkflowRun
from django_ai_sdk.workflows.runs import claim_run, execute_run
from tests.mocks.workflow import AGENT_GET, ENQUEUE, fake_agent, two_steps
from tests.oversight.helpers import race


def opened() -> str:
    return str(WorkflowRun.objects.create(workflow_definition=two_steps().model_dump()).pk)


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only")
class TestRacesOnPostgres:
    def test_one_of_many_claims_wins(self):
        run_id = opened()

        claims = race(*[lambda: async_to_sync(claim_run)(run_id)] * 8)

        assert sum(claim is not None for claim in claims) == 1
        run = WorkflowRun.objects.get(pk=run_id)
        assert (run.status, run.attempt) == ("running", 1)

    def test_many_deliveries_run_each_step_once(self):
        run_id = opened()
        model = fake_agent("ok")

        with patch(AGENT_GET, AsyncMock(return_value=model)):
            done = race(*[lambda: async_to_sync(execute_run)(run_id)] * 6)

        assert sum(result is not None for result in done) == 1
        assert model.run.await_count == 2
        run = WorkflowRun.objects.get(pk=run_id)
        assert (run.status, run.attempt) == ("completed", 1)
        assert list(StepRun.objects.filter(run_id=run_id).values_list("status", flat=True)) == [
            "completed",
            "completed",
        ]

    def test_two_resumes_at_once_queue_the_run_once(self):
        from django_ai_sdk.errors import UserError
        from django_ai_sdk.workflows.registry import register_workflow, reset_registry
        from django_ai_sdk.workflows.services import WorkflowService
        from tests.factories.db import UserFactory

        user = UserFactory.create()
        run_id = opened()
        # Keyed like a registered workflow: resuming an ad-hoc run also takes MANAGE.
        WorkflowRun.objects.filter(pk=run_id).update(
            status="failed", user=user, workflow_key="payroll"
        )
        register_workflow(two_steps().model_copy(update={"name": "payroll"}))

        def resume() -> Any:
            try:
                return async_to_sync(WorkflowService.resume)(run_id, user=user)
            except UserError as refused:
                return refused

        try:
            with patch(ENQUEUE) as enqueue:
                results = race(resume, resume)
        finally:
            reset_registry()

        assert sum(isinstance(result, WorkflowRun) for result in results) == 1
        assert enqueue.call_count == 1
        assert WorkflowRun.objects.get(pk=run_id).status == "pending"

    def test_a_sweep_and_a_beat_at_once_leave_one_owner(self):
        """Whichever lands first, the run is either the worker's or the sweep's, never both."""
        from django_ai_sdk.workflows.runs import RunRecorder
        from django_ai_sdk.workflows.steps import RunSuperseded
        from tests.mocks.workflow import FakeStep

        run_id = opened()
        run = async_to_sync(claim_run)(run_id)
        assert run is not None
        WorkflowRun.objects.filter(pk=run_id).update(
            heartbeat_at=timezone.now() - timedelta(hours=2)
        )
        recorder = RunRecorder(run, [FakeStep("first"), FakeStep("second")])

        def beat() -> Any:
            try:
                async_to_sync(recorder.beat)()
            except RunSuperseded as superseded:
                return superseded
            return "alive"

        beat_result, _ = race(beat, lambda: call_command("workflows_sweep", verbosity=0))

        status = WorkflowRun.objects.get(pk=run_id).status
        if beat_result == "alive":
            # The beat came first: the heartbeat is fresh, so the sweep found nothing.
            assert status == "running"
        else:
            assert status == "interrupted"
