"""Minimal Django settings for the SDK test suite.

Only what the tests need. Demo projects live in demos/ and carry their own settings;
nothing here should import from them.
"""

import os
from urllib.parse import unquote, urlsplit

SECRET_KEY = "tests-only-not-secret"
DEBUG = False
USE_TZ = True
# Django 6.0's default, which the migrations were made with; 5.2 still warns (W042) without it.
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django_tasks",
    "django_ai_sdk",
    "django_ai_sdk.tracing",
    "django_ai_sdk.workflows",
    "django_ai_sdk.oversight",
    "django_ai_sdk.oversight.authoring",
    "django_ai_sdk.integrations.mcp",
    "django_ai_sdk.integrations.weather",
    "tests.testapp",
]

AUTH_USER_MODEL = "testapp.User"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Opt-in PostgreSQL, for what SQLite cannot show (row locks, racing connections):
#   DATABASE_URL=postgres://postgres:pg@127.0.0.1:5432/postgres \
#       uv run --with "psycopg[binary]" pytest tests
if DATABASE_URL := os.environ.get("DATABASE_URL"):
    _url = urlsplit(DATABASE_URL)
    if _url.scheme not in {"postgres", "postgresql"}:
        raise ValueError(f"DATABASE_URL must be a postgres:// URL, got {_url.scheme!r}")
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": _url.path.lstrip("/") or "postgres",
            "USER": unquote(_url.username or ""),
            "PASSWORD": unquote(_url.password or ""),
            "HOST": _url.hostname or "",
            "PORT": str(_url.port or ""),
        }
    }

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]

ROOT_URLCONF = "tests.urls"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.SessionAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
}

TASKS = {"default": {"BACKEND": "django_tasks.backends.immediate.ImmediateBackend"}}
