from __future__ import annotations

from django.db import migrations, models
from django.utils.text import slugify


def backfill_slugs(apps, schema_editor):  # noqa: ANN001, ANN201
    """Give every existing row its own slug, or the unique index cannot be built."""
    WorkflowSettings = apps.get_model("django_ai_sdk_workflows", "WorkflowSettings")
    taken: set[str] = set()
    for row in WorkflowSettings.objects.order_by("created_at", "pk"):
        base = (slugify(row.name) or f"workflow-{str(row.pk)[:8]}")[:100]
        slug, suffix = base, 2
        while slug in taken:
            tail = f"-{suffix}"
            slug = f"{base[: 100 - len(tail)]}{tail}"
            suffix += 1
        taken.add(slug)
        row.slug = slug
        row.save(update_fields=["slug"])


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_workflows", "0002_workflow_run_inputs"),
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
        migrations.RemoveField(
            model_name="workflowrunstep",
            name="output_key",
        ),
        migrations.AddField(
            model_name="workflowsettings",
            name="slug",
            field=models.SlugField(default="", max_length=100),
            preserve_default=False,
        ),
        migrations.RunPython(backfill_slugs, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="workflowsettings",
            name="slug",
            field=models.SlugField(max_length=100, unique=True),
        ),
    ]
