from __future__ import annotations

import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="AutomationState",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("name", models.CharField(db_index=True, max_length=255, unique=True)),
                ("enabled", models.BooleanField(blank=True, null=True)),
                ("next_run_at", models.DateTimeField(db_index=True)),
                ("last_dispatched_at", models.DateTimeField(blank=True, null=True)),
                ("last_success_at", models.DateTimeField(blank=True, null=True)),
                ("locked_until", models.DateTimeField(blank=True, db_index=True, null=True)),
                ("schedule_repr", models.CharField(blank=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Automation",
                "verbose_name_plural": "Automations",
                "db_table": "django_ai_sdk_automation_state",
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="AutomationRun",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("name", models.CharField(db_index=True, max_length=255)),
                ("dispatch_id", models.UUIDField(db_index=True, default=uuid.uuid4)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("succeeded", "Succeeded"),
                            ("failed", "Failed"),
                            ("skipped", "Skipped"),
                        ],
                        db_index=True,
                        default="pending",
                        max_length=20,
                    ),
                ),
                (
                    "trigger",
                    models.CharField(
                        choices=[("schedule", "Schedule"), ("manual", "Manual")],
                        default="schedule",
                        max_length=20,
                    ),
                ),
                ("scheduled_for", models.DateTimeField(db_index=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("output", models.JSONField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
                ("skip_reason", models.CharField(blank=True, default="", max_length=255)),
                ("task_id", models.CharField(blank=True, max_length=64, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="automation_runs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "workflow_run",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="automation_runs",
                        to="django_ai_sdk.workflowrun",
                    ),
                ),
                (
                    "state",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="runs",
                        to="django_ai_sdk.automationstate",
                    ),
                ),
            ],
            options={
                "verbose_name": "Automation Run",
                "verbose_name_plural": "Automation Runs",
                "db_table": "django_ai_sdk_automation_runs",
                "ordering": ["-scheduled_for", "-created_at"],
                "indexes": [
                    models.Index(
                        fields=["name", "-scheduled_for"], name="django_ai_s_name_684517_idx"
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="AutomationSubscription",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("name", models.CharField(db_index=True, max_length=255)),
                ("enabled", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="automation_subscriptions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "django_ai_sdk_automation_subscriptions",
                "constraints": [
                    models.UniqueConstraint(
                        fields=("name", "user"), name="unique_automation_subscriber"
                    )
                ],
            },
        ),
    ]
