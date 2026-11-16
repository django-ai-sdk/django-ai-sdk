"""Judge classes by the dotted path settings name them with ("class" in JUDGES).

Imported lazily: lenses imports judges.base, and an eager import of classifier
or llm here would import the half-initialised lenses again.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.oversight.judges.base import Judge
    from django_ai_sdk.oversight.judges.classifier import Classifier
    from django_ai_sdk.oversight.judges.llm import LLMJudge
    from django_ai_sdk.oversight.judges.local import LocalClassifier
    from django_ai_sdk.oversight.judges.patterns import PatternJudge
    from django_ai_sdk.oversight.judges.typesafe import TypeSafe

_MODULES = {
    "Judge": "base",
    "Classifier": "classifier",
    "LLMJudge": "llm",
    "LocalClassifier": "local",
    "PatternJudge": "patterns",
    "TypeSafe": "typesafe",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)


__all__ = [
    "Classifier",
    "Judge",
    "LLMJudge",
    "LocalClassifier",
    "PatternJudge",
    "TypeSafe",
]
