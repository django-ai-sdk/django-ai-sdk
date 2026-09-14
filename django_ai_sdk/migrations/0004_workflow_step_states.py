"""Step outcomes the runner can record, and the slug a definition is named by."""

from __future__ import annotations

from typing import Any

from django.db import migrations, models
from django.utils.text import slugify


def backfill_slugs(apps: Any, schema_editor: Any) -> None:
    """Give every stored definition a unique registry key derived from its name."""
    WorkflowSettings = apps.get_model("django_ai_sdk", "WorkflowSettings")
    used: set[str] = set()
    for row in WorkflowSettings.objects.all().order_by("created_at"):
        base = (slugify(row.name) or f"workflow-{str(row.pk)[:8]}")[:100]
        slug, suffix = base, 2
        while slug in used:
            tail = f"-{suffix}"
            slug = f"{base[: 100 - len(tail)]}{tail}"
            suffix += 1
        used.add(slug)
        row.slug = slug
        row.save(update_fields=["slug"])


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0003_workflow_run_inputs"),
    ]

    operations = [
        migrations.AddField(
            model_name="workflowrunstep",
            name="detail",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AlterField(
            model_name="workflowrunstep",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("skipped", "Skipped"),
                ],
                default="pending",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="workflowsettings",
            name="slug",
            field=models.SlugField(blank=True, default="", max_length=100),
            preserve_default=False,
        ),
        migrations.RunPython(backfill_slugs, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="workflowsettings",
            name="slug",
            field=models.SlugField(max_length=100, unique=True),
        ),
    ]
