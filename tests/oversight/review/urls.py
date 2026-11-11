"""The oversight routes (review plus the installed opt-in apps) for the HTTP tests."""

from django.urls import path
from ninja import NinjaAPI
from ninja.security import SessionAuth

from django_ai_sdk.contrib import ninja as ai_sdk_routers

api = NinjaAPI(auth=SessionAuth(), urls_namespace="oversight-review-test")
ai_sdk_routers.register_error_handlers(api)
api.add_router("/", ai_sdk_routers.get_oversight_router())

# An API with no auth of its own: the router still signs readers in.
bare = NinjaAPI(urls_namespace="oversight-review-bare")
bare.add_router("/", ai_sdk_routers.get_oversight_router())

urlpatterns = [path("api/", api.urls), path("bare/", bare.urls)]
