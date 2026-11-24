"""Workflows on DBOS. Upgrades a 0.1.2a21 database in place; rows carry over."""

from __future__ import annotations

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models
from django.db.models import OuterRef, Subquery


def backfill(apps, schema_editor):  # noqa: ANN001, ANN201
    Workflow = apps.get_model("django_ai_sdk_workflows", "Workflow")
    WorkflowRun = apps.get_model("django_ai_sdk_workflows", "WorkflowRun")
    # "adhoc" now names runs of no stored workflow, so a stored one may not use it.
    for workflow in Workflow.objects.filter(slug="adhoc"):
        workflow.slug = f"adhoc-{str(workflow.pk)[:8]}"
        workflow.save(update_fields=["slug"])
    slug = Workflow.objects.filter(pk=OuterRef("workflow_id")).values("slug")[:1]
    WorkflowRun.objects.exclude(workflow=None).update(workflow_key=Subquery(slug))


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_workflows", "0003_workflow_step_states"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # Same tables (db_table is set): a rename keeps every row.
        migrations.RenameModel("WorkflowSettings", "Workflow"),
        migrations.RenameModel("WorkflowRunStep", "StepRun"),
        migrations.AlterModelOptions(
            name="steprun",
            options={
                "ordering": ["sequence"],
                "permissions": [("approve_steprun", "Can decide an approval step")],
                "verbose_name": "Step run",
                "verbose_name_plural": "Step runs",
            },
        ),
        migrations.AddField(
            model_name="steprun",
            name="turn_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="workflowrun",
            name="attempt",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="workflowrun",
            name="heartbeat_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="workflowrun",
            name="started_by",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="+",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="workflowrun",
            name="trigger",
            field=models.CharField(db_index=True, default="manual", max_length=120),
        ),
        migrations.AddField(
            model_name="workflowrun",
            name="workflow_key",
            field=models.CharField(db_index=True, default="adhoc", max_length=100),
        ),
        migrations.AlterField(
            model_name="steprun",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("skipped", "Skipped"),
                    ("waiting", "Waiting"),
                ],
                default="pending",
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name="workflowrun",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("cancelled", "Cancelled"),
                ],
                db_index=True,
                default="pending",
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name="workflowrun",
            name="task_id",
            field=models.CharField(blank=True, max_length=128, null=True),
        ),
        migrations.RunPython(backfill, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="workflow",
            constraint=models.CheckConstraint(
                condition=models.Q(("slug", "adhoc"), _negated=True),
                name="django_ai_sdk_workflow_slug_not_adhoc",
            ),
        ),
        migrations.AddIndex(
            model_name="workflowrun",
            index=models.Index(
                fields=["workflow_key", "trigger", "-created_at"],
                name="ai_sdk_wfrun_trigger_latest",
            ),
        ),
        migrations.AddIndex(
            model_name="workflowrun",
            index=models.Index(
                fields=["workflow", "-created_at"], name="ai_sdk_wfrun_workflow_latest"
            ),
        ),
        migrations.AddIndex(
            model_name="workflowrun",
            index=models.Index(fields=["user", "-created_at"], name="ai_sdk_wfrun_user_latest"),
        ),
    ]
