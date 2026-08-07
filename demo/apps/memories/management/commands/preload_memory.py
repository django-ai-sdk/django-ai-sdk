from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError
from django_ai_sdk.memories.models import Memory
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import ConflictError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Load memory entries from a folder on disk."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("slug", help="Memory slug to load entries into")
        parser.add_argument(
            "path",
            nargs="?",
            help="Relative path under AI_SDK_VECTOR_STORE_PATH (defaults to slug)",
        )

    def handle(self, *args: object, **options: object) -> None:
        asyncio.run(self._handle_async(*args, **options))

    async def _handle_async(self, *args: object, **options: object) -> None:
        slug = options["slug"]
        relative_path = options["path"] or slug

        try:
            memory = await Memory.objects.aget(slug=slug)
        except Memory.DoesNotExist as err:
            raise CommandError(f"Memory with slug '{slug}' not found.") from err

        User = get_user_model()
        user = await User.objects.filter(is_superuser=True).afirst()
        if user is None:
            raise CommandError("No superuser found.")

        from django.conf import settings

        base_path: str = getattr(settings, "AI_SDK_VECTOR_STORE_PATH", ".")
        folder = Path(base_path).parent / "knowledge" / str(relative_path)
        if not folder.is_dir():
            raise CommandError(f"Folder not found: {folder}")

        created_count = 0
        skipped_count = 0
        error_count = 0

        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue

            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            file = ContentFile(content, name=file_path.name)
            try:
                await MemoryService.upload_document(str(memory.id), file, user=user)
                created_count += 1
            except ConflictError:
                skipped_count += 1
                self.stdout.write(f"Skipped (duplicate): {file_path.name}")
            except Exception as e:  # noqa: BLE001
                error_count += 1
                self.stderr.write(f"Error ({file_path.name}): {e}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Memory '{slug}': "
                f"created {created_count}, "
                f"skipped {skipped_count}, "
                f"errors {error_count}."
            )
        )
