from contextlib import contextmanager


@contextmanager
def memory_permissions(*perm_paths):
    from django.test.utils import override_settings
    from django_ai_sdk.permissions import get_domain_permissions, PermissionDomain

    get_domain_permissions.cache_clear()
    with override_settings(AI_SDK_PERMISSIONS={"memory": list(perm_paths)}):
        get_domain_permissions.cache_clear()
        yield
    get_domain_permissions.cache_clear()


@contextmanager
def thread_permissions(*perm_paths):
    from django.test.utils import override_settings
    from django_ai_sdk.permissions import get_domain_permissions, PermissionDomain

    get_domain_permissions.cache_clear()
    with override_settings(AI_SDK_PERMISSIONS={"thread": list(perm_paths)}):
        get_domain_permissions.cache_clear()
        yield
    get_domain_permissions.cache_clear()
