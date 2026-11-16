from __future__ import annotations

import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class CreateModelIfMissing(migrations.CreateModel):
    """CreateModel that leaves an existing table alone.

    Up to 0.1.2a21 the core app's own 0001 created these tables in every project,
    with or without this app. A project that adds this app later has them already.
    """

    def database_forwards(self, app_label, schema_editor, from_state, to_state):  # noqa: ANN001, ANN201
        table = to_state.apps.get_model(app_label, self.name)._meta.db_table
        if table not in schema_editor.connection.introspection.table_names():
            super().database_forwards(app_label, schema_editor, from_state, to_state)


class Migration(migrations.Migration):
    """
    The workflow tables, as 0.1.2a21 shipped them; 0004 brings them up to date.

    A database that ran this in 0.1.2a21 or earlier does not run it again. One whose
    core app made the tables without this app installed gets their state only.
    """

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("django_ai_sdk", "0001_initial"),
    ]

    operations = [
        CreateModelIfMissing(
            name="WorkflowSettings",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("definition", models.JSONField(default=dict)),
                ("active", models.BooleanField(db_index=True, default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "created_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="workflows",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Workflow",
                "verbose_name_plural": "Workflows",
                "db_table": "django_ai_sdk_workflows",
                "ordering": ["name"],
            },
        ),
        CreateModelIfMissing(
            name="WorkflowRun",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        db_index=True,
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("workflow_definition", models.JSONField(blank=True, null=True)),
                ("input_messages", models.JSONField(default=list)),
                ("outputs", models.JSONField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
                ("task_id", models.CharField(blank=True, max_length=64, null=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="workflow_runs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "workflow",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="runs",
                        to="django_ai_sdk_workflows.workflowsettings",
                    ),
                ),
            ],
            options={
                "verbose_name": "Workflow Run",
                "verbose_name_plural": "Workflow Runs",
                "db_table": "django_ai_sdk_workflow_runs",
                "ordering": ["-created_at"],
            },
        ),
        CreateModelIfMissing(
            name="WorkflowRunStep",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("sequence", models.PositiveIntegerField()),
                ("step_name", models.CharField(blank=True, default="", max_length=255)),
                ("output_key", models.CharField(max_length=255)),
                ("output", models.JSONField(blank=True, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="steps",
                        to="django_ai_sdk_workflows.workflowrun",
                    ),
                ),
            ],
            options={
                "verbose_name": "Workflow Run Step",
                "verbose_name_plural": "Workflow Run Steps",
                "db_table": "django_ai_sdk_workflow_run_steps",
                "ordering": ["sequence"],
                "unique_together": {("run", "sequence")},
            },
        ),
    ]
