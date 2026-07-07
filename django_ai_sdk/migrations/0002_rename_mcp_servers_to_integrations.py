from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="assistantsettings",
            old_name="mcp_servers",
            new_name="integrations",
        ),
    ]
