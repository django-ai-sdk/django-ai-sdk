import os
import sys
from pathlib import Path

import environ

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
    # third-party
    "django_watchfiles",
    "django_ai_sdk",
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


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = "nl"

TIME_ZONE = "Europe/Amsterdam"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = "static/"


# LLM

OPENAI_API_KEY = env("OPENAI_API_KEY", default=None)
OPENAI_API_URL = env("OPENAI_API_URL", default=None)

# TODO: Implelement LLM configuration enums

AI_SDK_DEFAULT_MODEL = "zai-org/GLM-5-FP8"

# AI_SDK_DEFAULT_MODEL = "mistralai/Ministral-3-14B-Instruct-2512"
# AI_SDK_DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"
# AI_SDK_DEFAULT_MODEL = "openai/gpt-oss-120b"


# AI SDK Configuration
AI_SDK_ASSISTANTS = [
    "piratespeak.assistants.pirate_basic.PirateBasicAssistant",
    "piratespeak.assistants.pirate_openai.PirateOpenAIAssistant",
    "piratespeak.assistants.pirate_agent.PirateAgentAssistant",
    "piratespeak.assistants.agent_swarm.AgentSwarmAssistant",
]
