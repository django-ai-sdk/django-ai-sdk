from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="assistantsettings",
            name="supports_images",
            field=models.BooleanField(default=False),
        ),
    ]
