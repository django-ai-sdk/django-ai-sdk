"""Minimal Django settings for the SDK test suite.

Only what the tests need. Demo projects live in demos/ and carry their own settings;
nothing here should import from them.
"""

SECRET_KEY = "tests-only-not-secret"
DEBUG = False
USE_TZ = True

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django_tasks",
    "django_ai_sdk",
    "django_ai_sdk.tracing",
    "django_ai_sdk.workflows",
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

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]

ROOT_URLCONF = "tests.urls"

TASKS = {"default": {"BACKEND": "django_tasks.backends.immediate.ImmediateBackend"}}
