from __future__ import annotations

from django.db import migrations, models


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
            field=models.SlugField(default="", max_length=100, unique=True),
            preserve_default=False,
        ),
    ]
