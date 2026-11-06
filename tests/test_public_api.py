"""Each package's `__all__` is what it binds, what its docs import from it and what they describe."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Each package, and the pages that import it and describe every export.
PAGES = {
    "django_ai_sdk.oversight": ["oversight.md", "manual/oversight.md", "oversight-switches.md"],
    "django_ai_sdk.workflows": ["workflow-steps.md", "manual/workflows.md", "views-and-routing.md"],
}
IMPORT = re.compile(r"^[ \t]*from (django_ai_sdk[\w.]*) import (\([^)]*\)|[^\n]+)", re.MULTILINE)

by_package = pytest.mark.parametrize("package", list(PAGES))


def pages(package: str) -> list[str]:
    return [(ROOT / "docs/content" / page).read_text(encoding="utf-8") for page in PAGES[package]]


def doc_imports(package: str) -> list[tuple[str, str]]:
    """(module, name) for every import in the pages, the README and the package docstring."""
    texts = [*pages(package), (ROOT / "README.md").read_text(encoding="utf-8")]
    texts.append(importlib.import_module(package).__doc__ or "")
    found = []
    for text in texts:
        for match in IMPORT.finditer(text):
            statement = ast.parse(f"from {match.group(1)} import {match.group(2)}").body[0]
            assert isinstance(statement, ast.ImportFrom)
            found.extend((match.group(1), alias.name) for alias in statement.names)
    return found


@by_package
def test_everything_the_docs_import_exists_and_the_package_exports_it(package):
    found = doc_imports(package)
    assert package in {module for module, _ in found} | {f"{m}.{n}" for m, n in found}
    missing = []
    for module, name in found:
        if not hasattr(importlib.import_module(module), name):
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"{module}.{name}")
    assert missing == []
    exported = importlib.import_module(package).__all__
    assert {name for module, name in found if module == package} <= set(exported)


@by_package
def test_all_is_what_the_package_binds_and_the_docs_describe(package):
    module = importlib.import_module(package)
    bound = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and name != "annotations" and not inspect.ismodule(value)
    }
    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__) == bound
    text = "\n".join(pages(package))
    undocumented = [
        name
        for name in module.__all__
        if not re.search(rf"(?<![\w.]){re.escape(name)}(?=[(`.\s,\[])", text)
    ]
    assert undocumented == []
