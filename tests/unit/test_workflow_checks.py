"""The workflow system checks: registry entries that cannot be used, and the task backend."""

from __future__ import annotations

from typing import Any

import pytest
from django.core import checks
from django.test import override_settings
from django_tasks.backends.base import BaseTaskBackend

from django_ai_sdk.workflows.checks import check_workflows, check_workflows_deploy
from tests.oversight.helpers import check_ids as ids

STEP = "tests.mocks.workflow.FakeStep"
LISTENER = "tests.mocks.workflow.RecordingListener"
WORKER = {"default": {"BACKEND": f"{__name__}.WorkerBackend"}}


class WorkerBackend(BaseTaskBackend):
    """A backend with a worker somewhere else: tasks leave the request."""

    def enqueue(self, task: Any, args: Any, kwargs: Any) -> Any:
        raise NotImplementedError


class TestTheRegistries:
    @override_settings(AI_SDK_WORKFLOWS={"STEPS": {"ok": STEP}, "LISTENERS": {"ok": LISTENER}})
    def test_entries_that_import_are_fine(self):
        assert check_workflows() == []

    @override_settings(AI_SDK_WORKFLOWS={"STEPS": {"ocr": "nowhere.OcrStep"}})
    def test_a_step_that_does_not_import_is_e002(self):
        [error] = check_workflows()

        assert error.id == "ai_sdk.workflows.E002"
        assert "AI_SDK_WORKFLOWS['STEPS']['ocr']" in error.msg
        assert "does not import" in error.msg

    @override_settings(AI_SDK_WORKFLOWS={"STEPS": {"wrong": LISTENER}})
    def test_a_step_that_is_not_a_step_is_e002(self):
        [error] = check_workflows()

        assert error.id == "ai_sdk.workflows.E002"
        assert "not a Step subclass" in error.msg

    @override_settings(AI_SDK_WORKFLOWS={"STEPS": {"agent": STEP}})
    def test_a_step_under_the_reserved_agent_key_is_e004(self):
        """Definitions name agent steps "agent", so that class could never be reached."""
        [error] = check_workflows()

        assert error.id == "ai_sdk.workflows.E004"
        assert "reserved" in error.msg

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"gone": "nowhere.Listener", "wrong": STEP}})
    def test_a_listener_that_cannot_be_used_is_e003(self):
        assert ids(check_workflows()) == ["ai_sdk.workflows.E003"] * 2

    def test_they_are_registered_with_django(self):
        assert check_workflows in checks.registry.registry.get_checks()
        assert check_workflows_deploy in checks.registry.registry.get_checks(
            include_deployment_checks=True
        )
        assert check_workflows_deploy not in checks.registry.registry.get_checks()


class TestTheTaskBackend:
    @pytest.mark.parametrize(
        ("backend", "problem"),
        [
            ("django_tasks.backends.immediate.ImmediateBackend", "run inline"),
            ("django_tasks.backends.dummy.DummyBackend", "never run"),
        ],
    )
    def test_a_backend_without_a_worker_is_w001(self, backend, problem):
        with override_settings(TASKS={"default": {"BACKEND": backend}}):
            [warning] = check_workflows_deploy()

        assert warning.id == "ai_sdk.workflows.W001"
        assert problem in warning.msg

    def test_a_backend_with_a_worker_is_fine(self):
        with override_settings(TASKS=WORKER):
            assert check_workflows_deploy() == []
