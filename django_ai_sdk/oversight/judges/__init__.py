"""Judges answer a judge lens: patterns, classifiers and model judges.

Each is imported on first use to break an import cycle: lenses imports
judges.base, which loads this package, and an eager import of classifier or llm
here would import lenses again while it is half initialised. Helpers like
`base.resolve` stay in their modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.oversight.judges.base import Judge
    from django_ai_sdk.oversight.judges.classifier import Classifier
    from django_ai_sdk.oversight.judges.lexical import LexicalSupport
    from django_ai_sdk.oversight.judges.llm import LLMJudge
    from django_ai_sdk.oversight.judges.local import LocalClassifier
    from django_ai_sdk.oversight.judges.patterns import Patterns
    from django_ai_sdk.oversight.judges.typesafe import TypeSafe

# Each judge by the module that declares it.
_MODULES = {
    "Judge": "base",
    "Classifier": "classifier",
    "LexicalSupport": "lexical",
    "LLMJudge": "llm",
    "LocalClassifier": "local",
    "Patterns": "patterns",
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
    "LexicalSupport",
    "LocalClassifier",
    "Patterns",
    "TypeSafe",
]
