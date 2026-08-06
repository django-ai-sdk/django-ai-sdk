"""
Mock helpers for SDK testing.

When a real object isn't practical (side effects, singletons, external APIs),
these helpers create controlled fakes for unit tests.

    agent.py    — create_agent_mock(), create_mock_adapter_class(),
                      mock_agent_memories()
    registry.py     — patch_registry() context manager for the global registry
    storage.py      — mock_get_storage(), setup_thread_adapter(),
                      mock_thread_model()
    permissions.py  — memory_permissions() context manager
"""
