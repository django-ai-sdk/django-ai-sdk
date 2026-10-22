from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand

User = get_user_model()

_USERS = [
    {
        "email": "alice@example.com",
        "first_name": "Alice",
        "last_name": "Johnson",
    },
    {
        "email": "bob@example.com",
        "first_name": "Bob",
        "last_name": "Smith",
    },
    {
        "email": "carol@example.com",
        "first_name": "Carol",
        "last_name": "Williams",
    },
    {
        "email": "dave@example.com",
        "first_name": "Dave",
        "last_name": "Brown",
    },
    {
        "email": "eve@example.com",
        "first_name": "Eve",
        "last_name": "Davis",
    },
]

_GROUPS = [
    "Engineering",
    "Design",
    "Marketing",
    "Sales",
    "Legal",
]


class Command(BaseCommand):
    help = "Create demo users and groups"

    def handle(self, *args: object, **options: object) -> None:
        password = "password123"

        for group_name in _GROUPS:
            Group.objects.get_or_create(name=group_name)
            self.stdout.write(f"  Group: {group_name}")

        for data in _USERS:
            user, created = User.objects.get_or_create(
                email=data["email"],
                defaults={
                    "first_name": data["first_name"],
                    "last_name": data["last_name"],
                },
            )
            if created:
                user.set_password(password)
                user.save()
                self.stdout.write(f"  User: {data['email']}")
            else:
                self.stdout.write(f"  Skipped (exists): {data['email']}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Created {len(_USERS)} users and {len(_GROUPS)} groups. "
                f"Password for all users: {password}"
            )
        )
