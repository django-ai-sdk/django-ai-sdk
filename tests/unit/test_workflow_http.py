"""Ninja and DRF answer the workflow endpoints with the same JSON."""

from __future__ import annotations

from datetime import datetime

import pytest

from django_ai_sdk.workflows.models import StepRun, Workflow, WorkflowRun

DEFINITION = {"name": "parity", "steps": [{"name": "result", "agent_id": "a"}]}
ROOTS = {"ninja": "/api/workflows/", "drf": "/drf/workflows/"}


@pytest.fixture
def run(django_user_model):
    user = django_user_model.objects.create(email="owner@example.com")
    record = Workflow.objects.create(name="Parity", definition=DEFINITION)
    run = WorkflowRun.objects.create(
        workflow=record,
        workflow_key=record.slug,
        user=user,
        status=WorkflowRun.Status.COMPLETED,
        attempt=1,
        outputs={"result": "plain text"},
    )
    # A step may return any JSON, not only an object.
    StepRun.objects.create(
        run=run, sequence=0, step_name="result", status="completed", output="plain text"
    )
    StepRun.objects.create(
        run=run, sequence=1, step_name="notify", status="skipped", detail="result was skipped"
    )
    return run


def same_instant(value):
    """A timestamp as a datetime, so ISO spellings of one instant compare equal."""
    if isinstance(value, str) and value[:4].isdigit() and "T" in value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, list):
        return [same_instant(item) for item in value]
    if isinstance(value, dict):
        return {key: same_instant(item) for key, item in value.items()}
    return value


@pytest.mark.django_db
class TestParity:
    def get(self, client, run, path):
        client.force_login(run.user)
        return {
            name: same_instant(client.get(root + path.format(w=run.workflow_id, r=run.pk)).json())
            for name, root in ROOTS.items()
        }

    def test_a_run_reads_the_same_from_both(self, client, run):
        both = self.get(client, run, "{w}/runs/{r}/")

        assert both["ninja"] == both["drf"]
        assert both["ninja"]["steps"][0]["output"] == "plain text"
        # A skipped step says why over HTTP, not only in the ORM.
        assert both["ninja"]["steps"][1]["detail"] == "result was skipped"
        assert both["ninja"]["attempt"] == 1

    def test_the_run_list_reads_the_same_from_both(self, client, run):
        both = self.get(client, run, "{w}/runs/")

        assert both["ninja"] == both["drf"]

    def test_a_stored_workflow_reads_the_same_from_both(self, client, run):
        from tests.factories.db import MANAGE_WORKFLOWS, grant

        grant(run.user, MANAGE_WORKFLOWS)
        both = self.get(client, run, "{w}/")

        assert both["ninja"] == both["drf"]
        assert both["ninja"]["slug"] == "parity"

    def test_a_run_under_another_workflows_path_is_not_found_in_either(self, client, run):
        other = Workflow.objects.create(name="Other", definition=DEFINITION)
        client.force_login(run.user)

        for root in ROOTS.values():
            assert client.get(f"{root}{other.pk}/runs/{run.pk}/").status_code == 404

    def test_a_run_started_by_key_reads_the_same_from_both_by_its_id(self, client, run):
        # A registered or ad-hoc run has no stored workflow, so only its id finds it.
        keyed = WorkflowRun.objects.create(
            workflow_key="registered-parity",
            user=run.user,
            status=WorkflowRun.Status.FAILED,
            attempt=1,
            error="boom",
        )
        client.force_login(run.user)

        both = {
            name: same_instant(client.get(f"{root}runs/{keyed.pk}/").json())
            for name, root in ROOTS.items()
        }

        assert both["ninja"] == both["drf"]
        assert both["ninja"]["workflow_id"] is None
        assert both["ninja"]["workflow_key"] == "registered-parity"
        assert both["ninja"]["status"] == "failed"

    def test_someone_elses_run_by_id_is_not_found_in_either(self, client, run, django_user_model):
        stranger = django_user_model.objects.create(email="stranger@example.com")
        client.force_login(stranger)

        for root in ROOTS.values():
            assert client.get(f"{root}runs/{run.pk}/").status_code == 404
