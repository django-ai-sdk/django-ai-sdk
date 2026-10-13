---
title: Overriding Views
type: docs
weight: 6
---

The [ready-made endpoints](/views-and-routing/#pick-a-starting-point) are meant to be bent, not forked. Each recipe below changes one thing and keeps the rest of the SDK's behaviour, including permission checks, which live in the services and not in the views.

Rule of thumb: **extend, don't rewrite.** Call the SDK's own endpoint (Ninja) or `super()` (DRF) and adjust the result. Only drop down to the services when the response has to look completely different.

{{< callout type="info" >}}
Every recipe on this page runs in the test suite (`tests/unit/test_overriding_views.py`), so it stays correct as the SDK changes.
{{< /callout >}}

## django-ninja

`get_*_router()` returns a new `Router` on every call. Change it before you mount it.

### Add an endpoint

Decorate the returned router like any other:

```python
router = ai_sdk_routers.get_threads_router()

@router.get("/threads/{thread_id}/export/")
async def export_thread(request, thread_id: str):
    ...

api.add_router("/", router)
```

Or keep your endpoints in your own `Router` and mount it on the same `NinjaAPI`; the studio demo does this for its health, digest and account endpoints.

### Replace an endpoint

Exclude it by name, then register yours on the same path. Every SDK endpoint is an importable function, so you can call it and adjust its result:

```python
from django_ai_sdk.contrib.ninja import schemas, threads

router = ai_sdk_routers.get_threads_router(exclude={"list_threads"})

@router.get("/threads/", response=schemas.ThreadListResponse, operation_id="list_threads")
async def list_threads(request, limit: int = 100, offset: int = 0):
    response = await threads.list_threads(request, limit=limit, offset=offset)
    response.threads = [t for t in response.threads if t.message_count > 0]
    return response
```

- Names in `exclude` are the endpoint function names, which are also the OpenAPI `operationId`s. A name that doesn't exist raises `ValueError` at startup, so a typo can't silently leave the original in place.
- Pass the same `operation_id` and response schema (all of them live in `django_ai_sdk.contrib.ninja.schemas`) to keep a generated frontend client unchanged.
- Need a different response shape? Skip the wrapper and call the service (`ThreadService.threads(...)`) with your own schema.

{{< callout type="warning" >}}
Register the replacement on the router that `get_*_router(exclude=...)` returned, not on a separate `Router`: Ninja resolves a path per router, so a second router on a shared path answers `405 Method Not Allowed` for the other methods.
{{< /callout >}}

### Remove an endpoint

Exclude it and don't add a replacement:

```python
api.add_router("/", ai_sdk_routers.get_threads_router(exclude={"delete_all_threads"}))
```

### Change auth or throttling for a group of endpoints

`add_router()` overrides auth, throttling and tags per mount:

```python
from ninja.security import django_auth_superuser
from ninja.throttling import AuthRateThrottle

api.add_router("/", ai_sdk_routers.get_agents_router(), auth=django_auth_superuser)
api.add_router("/", ai_sdk_routers.get_workflows_router(), throttle=AuthRateThrottle("100/h"))
```

The same keywords can go to the factory instead (`ai_sdk_routers.get_agents_router(auth=...)`); they are passed straight to `Router(...)`.

### Change error responses

`register_error_handlers(api)` answers every exception with an [error code](/errors/). Ninja uses the most specific handler, so registering one for a narrower type overrides it for just that type:

```python
from django_ai_sdk.errors import NotFound

ai_sdk_routers.register_error_handlers(api)

@api.exception_handler(NotFound)
def not_found(request, exc):
    return api.create_response(request, {"error": "gone"}, status=404)
```

To change a code's message, status or `retryable` flag everywhere (chat streams included), use `register_error_code()` instead; see [Your own codes](/errors/#your-own-codes).

## Django REST framework

The viewsets are plain DRF: subclass one and register the subclass on your own router.

### Register your own router

`include("django_ai_sdk.contrib.drf.urls")` is all-or-nothing. Once you override a viewset, register the resources you want yourself:

```python
from rest_framework.routers import DefaultRouter

from django_ai_sdk.contrib.drf import MemoryViewSet, WorkflowViewSet
from django_ai_sdk.views.chat import ChatView

router = DefaultRouter()
router.register("threads", MyThreadViewSet, basename="thread")
router.register("memories", MemoryViewSet, basename="memory")
router.register("workflows", WorkflowViewSet, basename="workflow")

urlpatterns = [
    path("api/threads/<str:thread_id>/chat/", ChatView.as_view()),
    path("api/", include(router.urls)),
]
```

Viewsets use `thread_id`, `memory_id`, `workflow_id` and so on as their URL keyword, so a method's signature reads `def retrieve(self, request, thread_id=None)`.

### Change one action

Call `super()` and adjust the response:

```python
from django_ai_sdk.contrib.drf import ThreadViewSet

class MyThreadViewSet(ThreadViewSet):
    def list(self, request):
        response = super().list(request)
        response.data = [t for t in response.data if t["message_count"] > 0]
        return response
```

### Add an action

```python
from rest_framework.decorators import action
from rest_framework.response import Response

class MyThreadViewSet(ThreadViewSet):
    @action(detail=True, methods=["get"])
    def export(self, request, thread_id=None):
        return Response({"thread_id": thread_id, "format": "markdown"})
```

Inside an action, use DRF as usual: `self.get_serializer(data=request.data)` validates a body with the serializer registered for the action in `serializer_classes`, and `self.page(request)` returns `(limit, offset)` from the pagination class. Call the services' sync wrappers (`django_ai_sdk.memories.services.list_memories(...)` and friends), and raise service errors as they are; the viewset maps them.

### Swap a request serializer

Each viewset maps actions to request serializers in `serializer_classes`. Extend it with a stricter or larger serializer; the action itself stays the SDK's:

```python
from rest_framework import serializers

from django_ai_sdk.contrib.drf import MemoryViewSet
from django_ai_sdk.contrib.drf.serializers import MemorySerializer

class StrictMemorySerializer(MemorySerializer):
    name = serializers.CharField(min_length=3, max_length=80)

class MyMemoryViewSet(MemoryViewSet):
    serializer_classes = {**MemoryViewSet.serializer_classes, "create": StrictMemorySerializer}
```

Keep the field names the action reads (here `name`, `slug`, `description`, `is_public`); extra fields are yours to use in an overridden action.

### Change paging

List actions read `?limit=&offset=` through `pagination_class`, which defaults to `ApiPagination` (default 100, max 100):

```python
from django_ai_sdk.contrib.drf import ApiPagination, ThreadViewSet

class SmallPages(ApiPagination):
    default_limit = 20
    max_limit = 50

class MyThreadViewSet(ThreadViewSet):
    pagination_class = SmallPages
```

### Remove an action

For an extra action (anything declared with `@action`), set it to `None` and its route disappears:

```python
class MyThreadViewSet(ThreadViewSet):
    delete_all = None
```

For a standard method (`list`, `create`, `retrieve`, `partial_update`, `destroy`), refuse it:

```python
from rest_framework.exceptions import MethodNotAllowed

class MyThreadViewSet(ThreadViewSet):
    def destroy(self, request, thread_id=None):
        raise MethodNotAllowed("DELETE")
```

### Change auth, permissions or throttling

By default the viewsets use your `REST_FRAMEWORK` settings. Override per viewset like any DRF view:

```python
from rest_framework.permissions import IsAdminUser
from rest_framework.throttling import UserRateThrottle

class AdminWorkflowViewSet(WorkflowViewSet):
    permission_classes = [IsAdminUser]
    throttle_classes = [UserRateThrottle]
```

These classes are an extra gate in front of the SDK's own checks, not a replacement: a user who passes `IsAdminUser` still only sees the workflows the SDK's permission classes allow. To change *those* rules, configure `AI_SDK_PERMISSIONS` (see [Permissions](/manual/permissions/)).

### Change error responses

The viewsets answer with `exception_handler` from `get_exception_handler()`, whatever your `EXCEPTION_HANDLER` setting is. Override it on a subclass, usually by wrapping the SDK's:

```python
from django_ai_sdk.contrib.drf import exception_handler

def my_handler(exc, context):
    response = exception_handler(exc, context)
    if "ref" in response.data:  # an SDK error body, not one of DRF's own
        response["X-Error-Ref"] = response.data["ref"]
    return response

class MyThreadViewSet(ThreadViewSet):
    def get_exception_handler(self):
        return my_handler
```

## The chat view

`ChatView` takes an `agent` (for URLs without a thread, see [Your own views](/views-and-routing/#your-own-views)) and has two hooks:

```python
from django.http import JsonResponse

from django_ai_sdk.errors import error_response
from django_ai_sdk.views.chat import ChatView

class MyChatView(ChatView):
    async def get_agent(self, thread_id, user):
        agent = await super().get_agent(thread_id, user)
        # e.g. pick a variant, log, or refuse
        return agent

    def error_response(self, exc):
        status, body = error_response(exc)
        return JsonResponse({"error": body["code"]}, status=status)
```

For anything beyond that (custom auth, rate limiting, request logging), wrap it the Django way: decorators such as `method_decorator(ratelimit(...), name="post")`, or middleware.

The Ninja layer's chat endpoint is `add_message_to_thread` on the threads router; replace it like any other endpoint.
