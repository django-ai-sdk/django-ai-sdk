"""Who a queued run executes as: a worker has no request, so the user is reloaded."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from django.db.utils import OperationalError

from django_ai_sdk.tasks import aget_principal


@pytest.mark.django_db(transaction=True)
class TestAgetPrincipal:
    async def _user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_it_loads_the_user_the_run_was_created_for(self):
        user = await self._user()
        assert await aget_principal(user.pk) == user

    async def test_no_user_id_means_an_unowned_run(self):
        assert await aget_principal(None) is None

    # Reachable only when the user is deleted between reading the run and this lookup.
    async def test_a_user_deleted_since_the_enqueue_leaves_the_run_unowned(self, caplog):
        user = await self._user()
        pk = user.pk
        await user.adelete()

        with caplog.at_level("WARNING"):
            assert await aget_principal(pk, source="Workflow run 1") is None
        assert "no longer exists" in caplog.text

    async def test_a_database_failure_is_not_swallowed(self):
        from django.contrib.auth import get_user_model

        def boom(*args, **kwargs):
            raise OperationalError("connection gone")

        with patch.object(get_user_model().objects, "aget", boom), pytest.raises(OperationalError):
            await aget_principal(1)


@pytest.mark.django_db(transaction=True)
class TestQueuedWorkflowRunCarriesItsUser:
    async def test_the_executor_receives_the_run_s_user(self):
        from django_ai_sdk.workflows.models import WorkflowRun
        from django_ai_sdk.workflows.tasks import _execute_async
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        run = await WorkflowRun.objects.acreate(
            workflow_definition={
                "name": "w",
                "steps": [{"name": "s", "agent_id": "a", "output_key": "out"}],
            },
            input_messages=[{"id": "1", "role": "user", "content": "hi"}],
            user_id=user.pk,
        )

        executor = AsyncMock(return_value=({}, run))
        with patch("django_ai_sdk.workflows.executor.WorkflowExecutor.run", executor):
            await _execute_async(str(run.id))

        assert executor.await_args.kwargs["user"] == user

    async def test_an_unowned_run_reaches_the_executor_with_no_user(self):
        from django_ai_sdk.workflows.models import WorkflowRun
        from django_ai_sdk.workflows.tasks import _execute_async

        run = await WorkflowRun.objects.acreate(
            workflow_definition={
                "name": "w",
                "steps": [{"name": "s", "agent_id": "a", "output_key": "out"}],
            },
            input_messages=[{"id": "1", "role": "user", "content": "hi"}],
        )

        executor = AsyncMock(return_value=({}, run))
        with patch("django_ai_sdk.workflows.executor.WorkflowExecutor.run", executor):
            await _execute_async(str(run.id))

        assert executor.await_args.kwargs["user"] is None
