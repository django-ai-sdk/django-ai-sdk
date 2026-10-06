"""Settings every demo shares. A demo's own settings start with::

    BASE_DIR = Path(__file__).resolve().parent.parent
    environ.Env.read_env(BASE_DIR / ".env")  # before the import: DEBUG etc. come from it

    from demo_base.settings import *

and then add what depends on its own folder (database, storage, URLs) and its own apps,
agents and permissions.
"""

from __future__ import annotations

import environ
from corsheaders.defaults import default_headers

env = environ.Env(DEBUG=(bool, False))

# SECURITY WARNING: demo projects only; never deploy these settings.
SECRET_KEY = env(
    "SECRET_KEY", default="django-insecure-bu5o)x@7lydjdtfn92=mtwc4sobt=7(*-l)pc_s@-pqqnj97(2"
)
DEBUG = env("DEBUG")
ALLOWED_HOSTS: list[str] = []

if DEBUG:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"


INSTALLED_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # auth
    "allauth",
    "allauth.account",
    "allauth.headless",
    "allauth.usersessions",
    # third-party
    "corsheaders",
    "django_watchfiles",
    "rest_framework",
    "django_tasks",
    "django_tasks_db",
    # sdk
    "django_ai_sdk",
    "django_ai_sdk.tracing",
    "django_ai_sdk.workflows",
    "django_ai_sdk.integrations.mcp",
    "django_ai_sdk.integrations.weather",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "corsheaders.middleware.CorsMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Dev: ImmediateBackend runs tasks inline (no worker needed).
# DEBUG=False: DatabaseBackend, run `python manage.py db_worker`.
TASKS = {
    "default": {
        "BACKEND": (
            "django_tasks.backends.immediate.ImmediateBackend"
            if DEBUG
            else "django_tasks_db.DatabaseBackend"
        ),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Authentication: allauth headless, for the frontend in ../web on port 3000.
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]
HEADLESS_ONLY = True
HEADLESS_FRONTEND_URLS = {
    "account_confirm_email": "http://localhost:3000/verify-email/{key}",
    "account_reset_password_from_key": "http://localhost:3000/password/reset/key/{key}",
    "account_signup": "http://localhost:3000/signup",
}
CSRF_TRUSTED_ORIGINS = ["http://localhost:3000"]
ACCOUNT_USER_MODEL_USERNAME_FIELD = None
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
ACCOUNT_LOGIN_METHODS = {"email"}

CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
CORS_ALLOW_HEADERS = (*default_headers, "x-email-verification-key", "x-password-reset-key")
CORS_ALLOW_CREDENTIALS = True

LANGUAGE_CODE = "nl"
TIME_ZONE = "Europe/Amsterdam"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.SessionAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    # Error codes for every DRF view, not only the SDK's viewsets.
    "EXCEPTION_HANDLER": "django_ai_sdk.contrib.drf.exception_handler",
}


# AI SDK

OPENAI_API_KEY = env("OPENAI_API_KEY", default=None)
OPENAI_API_URL = env("OPENAI_API_URL", default=None)

# Default LLM model
AI_SDK_DEFAULT_MODEL = env("AI_SDK_DEFAULT_MODEL", default="openai/gpt-oss-120b")

# Dense RAG embeddings through the OpenAI-compatible API
AI_SDK_EMBEDDINGS_MODEL = env("AI_SDK_EMBEDDINGS_MODEL", default=None)
AI_SDK_EMBEDDINGS_DIM = env.int("AI_SDK_EMBEDDINGS_DIM", default=384)

# Relative to the working directory: each demo keeps its own vector stores.
AI_SDK_VECTOR_STORE_PATH = "stores/"

AI_SDK_ALLOWED_FILES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "text/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# MCP OAuth discovery (RFC 9728)
AI_SDK_MCP_DISCOVERY_TIMEOUT = 10  # seconds
AI_SDK_MCP_DISCOVERY_CACHE_TTL = 3600  # seconds
AI_SDK_MCP_OAUTH_SUCCESS_URL = "/settings/integrations"

# Integration caching, timeouts and circuit breaker (see
# django_ai_sdk.integrations.base.ResilientCache). Together these bound the worst case
# a slow or dead integration can add to a chat response.
AI_SDK_INTEGRATION_CACHE_TTL = 900  # seconds a discovered tool list stays fresh
AI_SDK_INTEGRATION_TIMEOUT = 3  # seconds; hard bound on a cache-miss fetch
AI_SDK_INTEGRATION_CB_COOLDOWN = 60  # seconds a failing integration is skipped

AI_SDK_TRACING_EXCLUDED_TAGS = [
    "haystack.agent.tools",
    "haystack.agent.state_schema",
]

# HuggingFace models to pre-download for offline embedding use
HF_PRELOAD_MODELS = [
    "Qdrant/bm42-all-minilm-l6-v2-attentions",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
