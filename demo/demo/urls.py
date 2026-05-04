"""
URL configuration for demo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django_ai_sdk.memories.views import router as memories_router
from ninja import NinjaAPI

from piratespeak.views import router as piratespeak_router

# Create the main API instance
api = NinjaAPI(title="Django AI SDK Demo", version="1.0.0")


api.add_router("/piratespeak", piratespeak_router)
api.add_router("/memories", memories_router)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api.urls),
]
