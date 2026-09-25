from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    """
    State-only: WorkflowRun/WorkflowRunStep/WorkflowSettings move to the
    `django_ai_sdk.workflows` app.
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
