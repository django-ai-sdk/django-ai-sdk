"""How the SDK and oversight's layers may import each other.

Core oversight is a layer of its own: it imports no workflows and no routes. The review layer
and authoring sit on it as opt-in parts, and each can be removed without touching the layers
below. The rest of the SDK reaches oversight only through the seams below.

Every import is read from the source, including lazy imports inside functions and imports
under `TYPE_CHECKING`, so a layer cannot lean on another one without it showing here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "django_ai_sdk"
OVERSIGHT = "django_ai_sdk.oversight"

# The only core modules the rest of the SDK may import, and why it needs each one.
SEAMS = {
    f"{OVERSIGHT}.agent_run": "the agent opens, runs and replays each turn through it",
    f"{OVERSIGHT}.agents": "a subagent finds the policy its agent declares in code",
    f"{OVERSIGHT}.conf": "the agent reads whether oversight is on, and its settings",
    f"{OVERSIGHT}.guard": "the adapter and subagents find the guard of the turn in flight",
    f"{OVERSIGHT}.policy": "agents and workflow steps accept a Policy",
    f"{OVERSIGHT}.plugin": "every Haystack Agent the SDK builds runs oversight as hooks from it",
    f"{OVERSIGHT}.registry": "an agent turns its Policy class into the policy's name",
    f"{OVERSIGHT}.stream": "the reply stream passes through oversight, and refusals stream",
    f"{OVERSIGHT}.types": "workflow steps and webhook tasks name the stage they run at",
}


@dataclass(frozen=True)
class Layer:
    modules: tuple[str, ...]
    """The packages and modules the layer is made of, with everything under them."""
    may_import: tuple[str, ...]
    """The other layers it may import. Every layer may import the rest of the SDK, and a layer
    that may import core may import its seams."""
    why: str
    avoids: tuple[str, ...] = ()
    """Packages outside every layer that it must not import either."""


# A module belongs to the first layer that names it, so core and the SDK, which are named by
# their whole package, take whatever the layers before them leave.
LAYERS = {
    "review": Layer(
        modules=(
            f"{OVERSIGHT}.services",
            f"{OVERSIGHT}.schemas",
            f"{OVERSIGHT}.permissions",
            # It narrows turns with the review layer's scope hooks.
            f"{OVERSIGHT}.admin",
        ),
        may_import=("core",),
        why="The read and review API over the record; turns run without it.",
    ),
    "review routes": Layer(
        modules=("django_ai_sdk.contrib.ninja.oversight",),
        may_import=("core", "review"),
        why="The HTTP routes that serve the review API.",
    ),
    "authoring": Layer(
        modules=(f"{OVERSIGHT}.authoring",),
        may_import=("core", "review"),
        why="Policies written at runtime, registered beside code's.",
    ),
    "authoring routes": Layer(
        modules=("django_ai_sdk.contrib.ninja.oversight_authoring",),
        may_import=("seams", "authoring"),
        why="The HTTP routes that serve authoring.",
    ),
    "core": Layer(
        modules=(OVERSIGHT,),
        may_import=(),
        why="Guards turns on its own: no layer above it, no workflows and no routes.",
        avoids=("django_ai_sdk.workflows", "django_ai_sdk.contrib"),
    ),
    "sdk": Layer(
        modules=("django_ai_sdk",),
        may_import=("seams", "review routes", "authoring routes"),
        why="The rest of the SDK: it uses oversight through its seams and mounts the routes.",
    ),
}


def is_under(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def layer_of(module: str) -> str | None:
    """The layer `module` belongs to, or None when it is not part of the SDK."""
    for name, layer in LAYERS.items():
        if any(is_under(module, part) for part in layer.modules):
            return name
    return None


def target_layer(module: str) -> str | None:
    """The layer an import of `module` reaches into, with the seams apart from the rest of core."""
    return "seams" if module in SEAMS else layer_of(module)


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_exists(module: str) -> bool:
    path = ROOT.joinpath(*module.split("."))
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


def imported_modules(path: Path) -> set[str]:
    """Every module `path` imports, with relative imports resolved. A `from x import y` counts
    as importing `x`, and `x.y` too when `y` is a submodule."""
    module = module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                target = ".".join([*base, *([node.module] if node.module else [])])
            else:
                target = node.module or ""
            found.add(target)
            found.update(
                f"{target}.{alias.name}"
                for alias in node.names
                if module_exists(f"{target}.{alias.name}")
            )
    return found


def sources(where: Path) -> list[Path]:
    return sorted(path for path in where.rglob("*.py") if "__pycache__" not in path.parts)


def forbidden_imports(path: Path) -> list[str]:
    name = layer_of(module_name(path))
    layer = LAYERS[name]
    allowed = {name, *layer.may_import, "sdk", None}
    if "core" in allowed:
        allowed.add("seams")
    return sorted(
        target
        for target in imported_modules(path)
        if target_layer(target) not in allowed
        or any(is_under(target, package) for package in layer.avoids)
    )


@pytest.mark.parametrize("path", sources(PACKAGE), ids=module_name)
def test_each_module_imports_only_what_its_layer_may(path):
    found = forbidden_imports(path)
    layer = layer_of(module_name(path))
    assert not found, f"{module_name(path)} ({layer}) may not import {found}"


def test_every_module_the_table_names_exists():
    named = [*SEAMS, *(module for layer in LAYERS.values() for module in layer.modules)]
    assert [module for module in named if not module_exists(module)] == []
    assert {layer_of(seam) for seam in SEAMS} == {"core"}


def test_the_scan_sees_lazy_and_relative_imports(tmp_path, monkeypatch):
    fake = tmp_path / "django_ai_sdk" / "oversight"
    fake.mkdir(parents=True)
    # Core must avoid workflows and contrib on every branch, so they make stable examples.
    (fake / "guard.py").write_text(
        "def f():\n    from django_ai_sdk.workflows import x\n    from ..contrib import y\n"
    )
    monkeypatch.setattr(f"{__name__}.ROOT", tmp_path)
    assert forbidden_imports(fake / "guard.py") == [
        "django_ai_sdk.contrib",
        "django_ai_sdk.workflows",
    ]
