"""
Mock helpers for SDK testing.

When a real object isn't practical (side effects, singletons, external APIs),
these helpers create controlled fakes for unit tests.

    assistant.py  — create_assistant_mock() for registry assistants
    registry.py   — patch_registry() context manager for the global registry
"""
