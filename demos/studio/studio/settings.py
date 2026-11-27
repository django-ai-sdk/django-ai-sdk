from __future__ import annotations

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

environ.Env.read_env(BASE_DIR / ".env")

from demo_base.settings import *

ROOT_URLCONF = "studio.urls"
ASGI_APPLICATION = "studio.asgi.application"
WSGI_APPLICATION = "studio.wsgi.application"

INSTALLED_APPS = [
    *INSTALLED_APPS,
    "studio.integrations.linear",
    "apps.users",
    "apps.agents",
    "apps.memories",
    "apps.integrations",
]

AUTH_USER_MODEL = "users.User"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    "django_ai_sdk": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": BASE_DIR},
    },
}


AI_SDK_AGENTS = [
    "apps.agents.pirate_basic.PirateBasicAgent",
    "apps.agents.agent_swarm.PirateSwarmAgent",
    "apps.agents.deep_research.DeepResearchAgent",
]

AI_SDK_RUNTIME_AGENT_BASES = [
    "apps.agents.runtime.DefaultRuntimeAgent",
]

AI_SDK_RUNTIME_AGENT_TOOLS = {
    "get_today": "apps.agents.tools.get_today",
    "get_memory_files": "apps.agents.tools.get_memory_files",
    "get_memory_file": "apps.agents.tools.get_memory_file",
}

AI_SDK_WORKFLOWS = {
    "STEPS": {
        "gather_thread": "apps.agents.steps.GatherStep",
        "thread_digest": "apps.agents.steps.DigestStep",
    },
    "LISTENERS": {
        "console_log": "apps.agents.listeners.ConsoleLogListener",
        "thread_message": "apps.agents.listeners.ThreadMessageListener",
    },
}

AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}

AI_SDK_VISION_MODEL = env("AI_SDK_VISION_MODEL", default=None)
AI_SDK_VISION_AGENT = "apps.agents.vision.VisionAgent" if AI_SDK_VISION_MODEL else None

AI_SDK_THREAD_FILE_URL = "api-1.0.0:download_thread_file"

AI_SDK_ALLOWED_FILES = {
    **AI_SDK_ALLOWED_FILES,
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

AI_SDK_INTEGRATIONS = {
    "linear": {"TOKEN": env("LINEAR_API_KEY", default="")},
}
