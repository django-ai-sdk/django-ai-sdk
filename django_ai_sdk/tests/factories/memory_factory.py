from factory.django import DjangoModelFactory
from django_ai_sdk.memories.models import Memory


class MemoryFactory(DjangoModelFactory):
    class Meta:
        model = Memory

    name = "Test Memory"
