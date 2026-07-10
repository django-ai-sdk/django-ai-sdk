"""Outbound client + health probe for the GitHub REST API."""

from __future__ import annotations

import httpx

_API_URL = "https://api.github.com"


def fetch_repos(token: str) -> list[dict]:
    """List the token's GitHub account's repositories (most recently pushed first)."""
    with httpx.Client(timeout=5, headers={"Authorization": f"Bearer {token}"}) as client:
        response = client.get(
            f"{_API_URL}/user/repos", params={"per_page": 10, "sort": "pushed"}
        )
        response.raise_for_status()
    return [{"name": r["full_name"], "url": r["html_url"]} for r in response.json()]


async def check_github_api() -> None:
    """Health probe for get_status(): confirm the public API responds (no auth needed)."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{_API_URL}/zen")
        response.raise_for_status()
