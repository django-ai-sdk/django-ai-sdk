# Generated migration for MCPServerConfig model

from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_mcp", "0002_mcp_oauth_client"),
    ]

    operations = [
        migrations.CreateModel(
            name="MCPServerConfig",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "name",
                    models.SlugField(
                        help_text="Registry key, e.g. 'zendesk'. Must be URL-safe.", unique=True
                    ),
                ),
                ("label", models.CharField(blank=True, max_length=200)),
                (
                    "hint",
                    models.TextField(
                        blank=True,
                        help_text=(
                            "What this server's data actually is, e.g. 'Company wiki, HR docs, "
                            "and engineering runbooks.' Prepended to every tool's description so "
                            "the model knows when to reach for it, not just that it exists."
                        ),
                    ),
                ),
                ("url", models.URLField()),
                (
                    "auth",
                    models.CharField(
                        choices=[
                            ("static", "No auth"),
                            ("token", "Shared token"),
                            ("oauth", "OAuth 2.1"),
                        ],
                        default="static",
                        max_length=20,
                    ),
                ),
                (
                    "token",
                    models.TextField(
                        blank=True, help_text="Encrypted. Only used when auth='token'."
                    ),
                ),
                ("client_id", models.CharField(blank=True, max_length=500)),
                (
                    "client_secret",
                    models.TextField(
                        blank=True, help_text="Encrypted. Only used when auth='oauth'."
                    ),
                ),
                ("scope", models.CharField(blank=True, max_length=500)),
                ("oauth_discovery_url", models.URLField(blank=True)),
                ("authorization_endpoint", models.URLField(blank=True)),
                ("token_endpoint", models.URLField(blank=True)),
                (
                    "tools",
                    models.JSONField(
                        blank=True, default=list, help_text="Tool allow-list; empty = all."
                    ),
                ),
                (
                    "enabled",
                    models.BooleanField(
                        default=True,
                        help_text="Unchecking removes it from the registry immediately.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "MCP server",
                "app_label": "django_ai_sdk_mcp",
            },
        ),
    ]
