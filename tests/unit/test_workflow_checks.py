from __future__ import annotations

from django.core import checks
from django.test import override_settings
from django_ai_sdk.workflows.checks import check_workflows
from tests.oversight.helpers import check_ids as ids

STEP = "tests.mocks.workflow.FakeStep"
LISTENER = "tests.mocks.workflow.RecordingListener"


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
        [error] = check_workflows()

        assert error.id == "ai_sdk.workflows.E004"
        assert "reserved" in error.msg

    @override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {"gone": "nowhere.Listener", "wrong": STEP}})
    def test_a_listener_that_cannot_be_used_is_e003(self):
        assert ids(check_workflows()) == ["ai_sdk.workflows.E003"] * 2

    def test_they_are_registered_with_django(self):
        assert check_workflows in checks.registry.registry.get_checks()
