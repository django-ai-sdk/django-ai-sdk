"""
Django model factories for database-backed tests.
"""

from asgiref.sync import sync_to_async

import factory
from django.contrib.auth import get_user_model
from factory.django import DjangoModelFactory
from factory.declarations import Trait, SubFactory
from factory.faker import Faker
from django_ai_sdk.memories.models import Memory, MemoryUser


class AsyncFactoryMixin:
    """Mixin adding async create support to factory_boy DjangoModelFactory."""

    @classmethod
    async def acreate(cls, **kwargs):
        return await sync_to_async(cls.create)(**kwargs)


class UserFactory(AsyncFactoryMixin, DjangoModelFactory):
    class Meta:
        model = get_user_model()
        django_get_or_create = ("email",)

    email = factory.Sequence(lambda n: f"user{n}@example.com")

    class Params:
        staff = Trait(is_staff=True)
        superuser = Trait(is_superuser=True)


class MemoryFactory(AsyncFactoryMixin, DjangoModelFactory):
    class Meta:
        model = Memory

    name = Faker("sentence", nb_words=3)
    is_public = True

    class Params:
        private = Trait(is_public=False)


class MemoryUserFactory(AsyncFactoryMixin, DjangoModelFactory):
    class Meta:
        model = MemoryUser

    user = SubFactory(UserFactory)
    memory = SubFactory(MemoryFactory)
    can_manage = False

    class Params:
        manager = Trait(can_manage=True)
