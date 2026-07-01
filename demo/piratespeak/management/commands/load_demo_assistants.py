from django.conf import settings
from django.core.management.base import BaseCommand
from django_ai_sdk.assistants.models import AssistantSettings

_MODEL = getattr(settings, "AI_SDK_DEFAULT_MODEL", "gpt-4o")

_ASSISTANTS = [
    {
        "name": "General Assistant",
        "slug": "general",
        "system_prompt": (
            "You are a helpful, friendly assistant. Answer questions clearly and concisely."
        ),
        "tools": ["get_today"],
        "suggestion_enabled": True,
        "title_generation": True,
    },
    {
        "name": "Research Assistant",
        "slug": "research",
        "system_prompt": (
            "You are a research assistant with access to the company's knowledge base. "
            "Ground your answers in the provided documents and cite your sources."
        ),
        "tools": ["get_today"],
        "memories": ["research-papers", "product-specifications"],
        "suggestion_enabled": True,
        "title_generation": True,
    },
    {
        "name": "Staff Assistant",
        "slug": "staff",
        "system_prompt": (
            "You are an internal assistant for staff members. "
            "You have access to legal documents and meeting notes."
        ),
        "tools": ["get_today"],
        "memories": ["legal-documents", "meeting-notes"],
        "suggestion_enabled": False,
        "title_generation": True,
    },
    {
        "name": "Disabled Assistant",
        "slug": "disabled-example",
        "system_prompt": "This assistant is intentionally disabled.",
        "suggestion_enabled": False,
        "title_generation": False,
        "active": False,
    },
]


class Command(BaseCommand):
    help = "Seed demo runtime assistants"

    def handle(self, *args: object, **options: object) -> None:
        AssistantSettings.objects.all().delete()
        self.stdout.write("Deleted existing runtime assistants.")

        base_class = ""
        bases = getattr(settings, "AI_SDK_RUNTIME_ASSISTANT_BASES", [])
        if bases:
            cls_path = bases[0]
            parts = cls_path.rsplit(".", 1)
            if len(parts) == 2:
                from django.utils.module_loading import import_string

                try:
                    cls = import_string(cls_path)
                    base_class = f"{cls.__module__}.{cls.__qualname__}"
                except ImportError:
                    base_class = ""

        for data in _ASSISTANTS:
            AssistantSettings.objects.create(
                name=data["name"],
                slug=data["slug"],
                assistant=base_class,
                model=_MODEL,
                system_prompt=data.get("system_prompt", ""),
                tools=data.get("tools", []),
                mcp_servers=data.get("mcp_servers", []),
                memories=data.get("memories", []),
                users=data.get("users", []),
                groups=data.get("groups", []),
                suggestion_enabled=data.get("suggestion_enabled", False),
                title_generation=data.get("title_generation", True),
                file_upload=data.get("file_upload", False),
                active=data.get("active", True),
            )
            self.stdout.write(f"  Created: {data['name']}")

        self.stdout.write(self.style.SUCCESS(f"Created {len(_ASSISTANTS)} demo assistants."))
