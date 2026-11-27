from __future__ import annotations

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django_ai_sdk.permissions import Operation, WorkflowDefaultPermission
from django_ai_sdk.workflows.models import WorkflowRun

from apps.agents.permissions import StudioWorkflowPermission


@pytest.mark.django_db
def test_an_approver_can_decide_on_their_own_run_in_the_studio() -> None:
    approver = get_user_model().objects.create_superuser(email="approver@example.com", password="x")
    own_run = WorkflowRun(user_id=approver.pk, started_by_id=approver.pk)

    def allowed(permission) -> bool:
        return async_to_sync(permission.has_object_permission)(
            approver, Operation.APPROVE_WORKFLOW, own_run
        )

    assert allowed(StudioWorkflowPermission())
    assert not allowed(WorkflowDefaultPermission())
