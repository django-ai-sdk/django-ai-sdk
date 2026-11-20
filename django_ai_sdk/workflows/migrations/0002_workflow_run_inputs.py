from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_workflows", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="workflowrun",
            name="inputs",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.RemoveField(
            model_name="workflowrun",
            name="input_messages",
        ),
    ]
