from __future__ import annotations

import os
import sys
from pathlib import Path

import environ
from corsheaders.defaults import default_headers

env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Take environment variables from .env file
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

# Add the parent directory to Python path to import django_ai_sdk
sys.path.insert(0, str(BASE_DIR.parent))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/6.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-bu5o)x@7lydjdtfn92=mtwc4sobt=7(*-l)pc_s@-pqqnj97(2"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG")

if DEBUG:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

ALLOWED_HOSTS = []


# Application definition

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
    # mcp integration
    "django_ai_sdk.integrations.mcp",
    # default integrations
    "piratespeak.integrations.linear",
    "django_ai_sdk.integrations.weather",
    # local
    "piratespeak",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # auth
    "allauth.account.middleware.AccountMiddleware",
    "corsheaders.middleware.CorsMiddleware",
]

ROOT_URLCONF = "demo.urls"

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

# ASGI application
ASGI_APPLICATION = "demo.asgi.application"


# Background tasks
# Dev: ImmediateBackend runs tasks inline (no worker needed).
# Prod (DEBUG=False): DatabaseBackend — run `python manage.py db_worker`.
TASKS = {
    "default": {
        "BACKEND": (
            "django_tasks.backends.immediate.ImmediateBackend"
            if DEBUG
            else "django_tasks_db.DatabaseBackend"
        ),
    }
}

# Database
# https://docs.djangoproject.com/en/6.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Authentication
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

# Accounts
ACCOUNT_USER_MODEL_USERNAME_FIELD = None
ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
ACCOUNT_LOGIN_METHODS = {"email"}


# CORS

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]

CORS_ALLOW_HEADERS = (
    *default_headers,
    "x-email-verification-key",
    "x-password-reset-key",
)

CORS_ALLOW_CREDENTIALS = True

# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = "nl"

TIME_ZONE = "Europe/Amsterdam"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = "static/"


# AI SDK Configuration

OPENAI_API_KEY = env("OPENAI_API_KEY", default=None)
OPENAI_API_URL = env("OPENAI_API_URL", default=None)

# Default model
# AI_SDK_DEFAULT_MODEL = "google/gemma-4-31B-it"
# AI_SDK_DEFAULT_MODEL = "zai-org/GLM-5.1-FP8"
# AI_SDK_DEFAULT_MODEL = "mistralai/Ministral-3-14B-Instruct-2512"
# AI_SDK_DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"
AI_SDK_DEFAULT_MODEL = "openai/gpt-oss-120b"

# Base classes available for runtime configured assistants
AI_SDK_RUNTIME_ASSISTANT_BASES = [
    "piratespeak.assistants.runtime.DefaultRuntimeAssistant",
]

# Tools selectable in runtime assistant configuration
AI_SDK_RUNTIME_ASSISTANT_TOOLS = {
    "get_today": "piratespeak.assistants.tools.get_today",
    "get_memory_files": "piratespeak.assistants.tools.get_memory_files",
}

# Default Workflow actions
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "piratespeak.actions.ConsoleLogAction",
}

# Default asssitants
AI_SDK_ASSISTANTS = [
    "piratespeak.assistants.pirate_basic.PirateBasicAssistant",
    "piratespeak.assistants.agent_swarm.AgentSwarmAssistant",
]

# Permission overrides by domain
AI_SDK_PERMISSIONS = {
    "memory": [
        "piratespeak.permissions.AllowAnonymousMemoryPermission",
    ],
}


# Default vector store path
AI_SDK_VECTOR_STORE_PATH = "stores/"


# Integrations are Django apps (see INSTALLED_APPS above) that register themselves on
# ready(). INSTALLED_APPS decides which exist; this dict configures them, keyed by
# integration name, in the same shape as DATABASES or CACHES. A missing credential
# doesn't crash boot: the integration reports that it needs setup instead. `weather`
# needs none at all, so it isn't listed here and still works out of the box.
AI_SDK_INTEGRATIONS = {
    "linear": {"TOKEN": env("LINEAR_API_KEY", default="")},
}


# MCP OAuth discovery (RFC 9728)
AI_SDK_MCP_DISCOVERY_TIMEOUT = 10  # seconds
AI_SDK_MCP_DISCOVERY_CACHE_TTL = 3600  # seconds (1 hour)
AI_SDK_MCP_OAUTH_SUCCESS_URL = "/settings/integrations"

# Integration caching, timeouts and circuit breaker (see
# django_ai_sdk.integrations.base.ResilientCache). Together these bound the worst case
# a slow or dead integration can add to a chat response.
AI_SDK_INTEGRATION_CACHE_TTL = 900  # seconds a discovered tool list stays fresh
AI_SDK_INTEGRATION_TIMEOUT = 3  # seconds; hard bound on a cache-miss fetch
AI_SDK_INTEGRATION_CB_COOLDOWN = 60  # seconds a failing integration is skipped


REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "EXCEPTION_HANDLER": "piratespeak.exceptions.api_exception_handler",
}

# Allowed upload filetypes
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
