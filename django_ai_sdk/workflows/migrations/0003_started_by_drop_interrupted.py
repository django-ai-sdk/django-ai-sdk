from __future__ import annotations

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_workflows", "0002_dbos_runs"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
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
        # DBOS recovers a dead worker's runs, so nothing marks a run interrupted any
        # more. Ones from before resume like any failed run.
        migrations.RunSQL(
            "UPDATE django_ai_sdk_workflow_runs SET status = 'failed' WHERE status = 'interrupted'",
            reverse_sql=migrations.RunSQL.noop,
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
    ]
