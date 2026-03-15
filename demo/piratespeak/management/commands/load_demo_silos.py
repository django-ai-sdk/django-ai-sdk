from django.core.management.base import BaseCommand
from django_ai_sdk.silos.models import Silo


class Command(BaseCommand):
    help = "Add demo silos for testing"

    def handle(self, *args: object, **options: object) -> None:
        Silo.objects.all().delete()
        self.stdout.write("Deleted existing silos and documents.")

        silos_data = [
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

        for silo_data in silos_data:
            Silo.objects.create(name=silo_data["name"], description=silo_data["description"])

        self.stdout.write(self.style.SUCCESS(f"Created {len(silos_data)} silos."))
