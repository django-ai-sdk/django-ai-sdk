from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    """State-only: WorkflowRun/WorkflowRunStep/WorkflowSettings move to the
    `django_ai_sdk_workflows` app (see that app's 0001_initial, which state-creates
    them there in the same shape 0001_initial left them in here). `db_table` never
    changed, so existing tables and rows are untouched -- only migration state moves.
    """

    dependencies = [
        ("django_ai_sdk", "0002_memory_metadata_storage"),
        ("django_ai_sdk_workflows", "0001_initial"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.DeleteModel(name="WorkflowRunStep"),
                migrations.DeleteModel(name="WorkflowRun"),
                migrations.DeleteModel(name="WorkflowSettings"),
            ],
        ),
    ]
