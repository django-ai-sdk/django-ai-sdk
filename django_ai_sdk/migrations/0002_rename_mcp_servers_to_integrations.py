from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    """Rename ``mcp_servers`` to ``integrations``.

    A pure rename: the field stays ``JSONField(default=list)``, so existing values
    (``["notion", "linear"]``) remain valid as-is and no data migration is needed.
    """

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
