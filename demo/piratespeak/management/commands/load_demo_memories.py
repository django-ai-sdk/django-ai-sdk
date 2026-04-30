from django.core.management.base import BaseCommand
from django_ai_sdk.memories.models import Memory


class Command(BaseCommand):
    help = "Add demo memories for testing"

    def handle(self, *args: object, **options: object) -> None:
        Memory.objects.all().delete()
        self.stdout.write("Deleted existing memories and documents.")

        memories_data = [
            {
                "name": "Legal Documents",
                "description": "Contracts, NDAs, and legal agreements for the company.",
            },
            {
                "name": "Product Specifications",
                "description": "Product requirements, roadmaps, and technical specs.",
                "documents": [],
            },
            {
                "name": "Meeting Notes",
                "description": "Records from team meetings, standups, and planning sessions.",
                "documents": [],
            },
            {
                "name": "Research Papers",
                "description": "Research findings, market analysis, and white papers.",
                "documents": [],
            },
        ]

        for memory_data in memories_data:
            Memory.objects.create(name=memory_data["name"], description=memory_data["description"])

        self.stdout.write(self.style.SUCCESS(f"Created {len(memories_data)} memories."))
