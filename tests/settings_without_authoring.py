from tests.settings import *

INSTALLED_APPS = [app for app in INSTALLED_APPS if app != "django_ai_sdk.oversight.authoring"]
