# Generated migration for MCPOAuthClient model

from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_mcp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="MCPOAuthClient",
            fields=[
                (
                    "server_name",
                    models.CharField(max_length=100, primary_key=True, serialize=False),
                ),
                ("client_id", models.CharField(max_length=500)),
                ("client_secret", models.TextField(blank=True)),
                ("redirect_uri", models.URLField()),
                ("registered_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "app_label": "django_ai_sdk_mcp",
            },
        ),
    ]
