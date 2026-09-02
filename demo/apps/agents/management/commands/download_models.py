from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from django.core.management.base import BaseCommand
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Download HuggingFace models to HF_HOME for offline use."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--models",
            nargs="*",
            help="Specific model repo IDs to download (defaults to HF_PRELOAD_MODELS setting)",
        )

    def handle(self, *args: object, **options: object) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError

        models: list[Any] = cast("list[Any]", options.get("models")) or list(
            resolve_setting("HF_PRELOAD_MODELS", [])
        )
        if not models:
            self.stdout.write("No models to download (HF_PRELOAD_MODELS is empty)")
            return

        for entry in models:
            if isinstance(entry, str):
                repo_id = entry
                kwargs = {}
            else:
                repo_id = entry["repo_id"]
                kwargs = {k: v for k, v in entry.items() if k != "repo_id"}

            self.stdout.write(f"Downloading {repo_id}...")
            try:
                path = snapshot_download(repo_id, **kwargs)
                self.stdout.write(self.style.SUCCESS(f"  → {path}"))
            except (OSError, HfHubHTTPError) as e:
                self.stderr.write(self.style.ERROR(f"  Failed {repo_id}: {e}"))
