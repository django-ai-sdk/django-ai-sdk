"""Vault exceptions."""

from __future__ import annotations


class VaultError(Exception):
    """The vault cannot encrypt or decrypt as asked."""


class ScopeShredded(VaultError):
    """The scope was shredded, so nothing new can be stored under it."""


class KeyMismatch(VaultError):
    """The key does not decrypt this. It is a different or retired key, or the bytes changed.

    Unlike ScopeShredded, the data may still decrypt with the key that wrapped it.
    """


__all__ = ["ScopeShredded", "KeyMismatch", "VaultError"]
