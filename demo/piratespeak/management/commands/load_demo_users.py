from __future__ import annotations

from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand

_USERS = [
    {
        "username": "alice",
        "email": "alice@example.com",
        "first_name": "Alice",
        "last_name": "Johnson",
    },
    {
        "username": "bob",
        "email": "bob@example.com",
        "first_name": "Bob",
        "last_name": "Smith",
    },
    {
        "username": "carol",
        "email": "carol@example.com",
        "first_name": "Carol",
        "last_name": "Williams",
    },
    {
        "username": "dave",
        "email": "dave@example.com",
        "first_name": "Dave",
        "last_name": "Brown",
    },
    {
        "username": "eve",
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
                username=data["username"],
                defaults={
                    "email": data["email"],
                    "first_name": data["first_name"],
                    "last_name": data["last_name"],
                },
            )
            if created:
                user.set_password(password)
                user.save()
                self.stdout.write(f"  User: {data['username']} ({data['email']})")
            else:
                self.stdout.write(f"  Skipped (exists): {data['username']}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Created {len(_USERS)} users and {len(_GROUPS)} groups. "
                f"Password for all users: {password}"
            )
        )
