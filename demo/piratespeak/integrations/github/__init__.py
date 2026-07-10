"""Demo API-backed integration: list a GitHub account's repos via a *factory* tool.

Demonstrates the factory ``APIIntegration.tools`` shape: a callable that builds
the tool per request instead of a static ready-made one — the pattern a real
integration uses to carry a credential (a per-user token, an API key) into the
tool it hands the assistant. Needs a `GITHUB_TOKEN` env var (a plain PAT is
enough) to enable.
"""
