"""
Permission-related mock helpers.

``memory_permissions()`` context manager handles the ``override_settings``
+ ``cache_clear()`` dance needed when switching memory permission classes.
"""

from contextlib import contextmanager


@contextmanager
def memory_permissions(*perm_paths):
    """Context manager that sets ``AI_SDK_MEMORY_PERMISSIONS`` and clears caches.

    Usage::

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.delete_memory(mem_id, user=user)
    """
    from django.test.utils import override_settings
    from django_ai_sdk.memories.services import _get_memory_permissions

    _get_memory_permissions.cache_clear()
    with override_settings(AI_SDK_MEMORY_PERMISSIONS=list(perm_paths)):
        _get_memory_permissions.cache_clear()
        yield
    _get_memory_permissions.cache_clear()
