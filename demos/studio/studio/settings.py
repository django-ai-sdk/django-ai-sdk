"""Studio: the full-featured demo. Shared settings come from ``demo_base``."""

from __future__ import annotations

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

# Before the shared settings: they read DEBUG, API keys, ... from the environment.
environ.Env.read_env(BASE_DIR / ".env")

from demo_base.settings import *  # noqa: E402

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
    # Uploaded documents and thread files, inside this demo's folder.
    "django_ai_sdk": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": BASE_DIR},
    },
}


# Agents

AI_SDK_AGENTS = [
    "apps.agents.pirate_basic.PirateBasicAgent",
    "apps.agents.agent_swarm.PirateSwarmAgent",
    "apps.agents.deep_research.DeepResearchAgent",
]

# Base classes available for runtime configured agents
AI_SDK_RUNTIME_AGENT_BASES = [
    "apps.agents.runtime.DefaultRuntimeAgent",
]

# Tools selectable in runtime agent configuration
AI_SDK_RUNTIME_AGENT_TOOLS = {
    "get_today": "apps.agents.tools.get_today",
    "get_memory_files": "apps.agents.tools.get_memory_files",
    "get_memory_file": "apps.agents.tools.get_memory_file",
}

# Workflow composables. The package ships neither step types nor actions; both
# registries are the gate on what a JSON definition may name.
AI_SDK_WORKFLOW_STEPS = {
    "gather_thread": "apps.agents.steps.GatherStep",
    "thread_digest": "apps.agents.steps.DigestStep",
}
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
    "thread_message": "apps.agents.actions.ThreadMessageAction",
}

# Permission overrides by domain
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}

# Files and images

# Answers image questions (ask_image) and captions uploaded images. Agents on this model
# get image attachments as pixels; others get the caption.
AI_SDK_VISION_MODEL = env("AI_SDK_VISION_MODEL", default=None)

# The thread-file download view (storage isn't publicly served); file parts in the chat
# link to it. Ninja names routes "<namespace>:<view function>".
AI_SDK_THREAD_FILE_URL_NAME = "api-1.0.0:download_thread_file"

# The shared text types, plus what the studio's PDF and image pipelines read.
AI_SDK_ALLOWED_FILES = {
    **AI_SDK_ALLOWED_FILES,
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Integrations are Django apps (see INSTALLED_APPS) that register themselves on ready().
# This dict configures them by name. A missing credential doesn't crash boot: the
# integration reports that it needs setup instead. `weather` needs none at all.
AI_SDK_INTEGRATIONS = {
    "linear": {"TOKEN": env("LINEAR_API_KEY", default="")},
}
