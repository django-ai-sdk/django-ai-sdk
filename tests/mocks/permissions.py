"""
Permission-related mock helpers.

``memory_permissions()`` context manager handles the ``override_settings``
+ ``cache_clear()`` dance needed when switching memory permission classes.
"""

from contextlib import contextmanager


@contextmanager
def memory_permissions(*perm_paths):
    """Context manager that sets ``AI_SDK_PERMISSIONS`` and clears caches.

    Usage::

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.delete_memory(mem_id, user=user)
    """
    from django.test.utils import override_settings
    from django_ai_sdk.permissions import get_domain_permissions, PermissionDomain

    get_domain_permissions.cache_clear()
    with override_settings(AI_SDK_PERMISSIONS={"memory": list(perm_paths)}):
        get_domain_permissions.cache_clear()
        yield
    get_domain_permissions.cache_clear()


@contextmanager
def thread_permissions(*perm_paths):
    """Context manager that sets ``AI_SDK_PERMISSIONS`` for thread domain and clears caches.

    Usage::

        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
            result = await ThreadService.create_thread(assistant_id="x", user=None)
    """
    from django.test.utils import override_settings
    from django_ai_sdk.permissions import get_domain_permissions, PermissionDomain

    get_domain_permissions.cache_clear()
    with override_settings(AI_SDK_PERMISSIONS={"thread": list(perm_paths)}):
        get_domain_permissions.cache_clear()
        yield
    get_domain_permissions.cache_clear()
