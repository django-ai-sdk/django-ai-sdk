from __future__ import annotations

from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import AnonymousUser

type UserType = AbstractBaseUser | AnonymousUser | None
type RequiredUser = AbstractBaseUser | AnonymousUser
