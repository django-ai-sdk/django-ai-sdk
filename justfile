setup_extras := "--extra qdrant --extra mcp --extra files --extra dbos"
# Leaves out local: its onnxruntime stack is heavy, and the tests fake the model.
test_extras := setup_extras + " --extra chroma --extra providers --extra ninja --extra drf --extra typesafe"

# Override on the command line or from the environment, for example:
#   just UV_RUN_ARGS='--with "django==5.2.*"' TEST_PATHS=tests/oversight test
UV_RUN_ARGS := env("UV_RUN_ARGS", "")
TEST_PATHS := env("TEST_PATHS", "tests")
PYTEST_ARGS := env("PYTEST_ARGS", "")

# List the recipes
default:
    @just --list

# Install the dev environment and the git hooks
setup:
    uv sync {{setup_extras}}
    uv run lefthook install

# Install every extra, including the torch-based transformers stack
setup-all:
    uv sync --all-extras
    uv run lefthook install

# Format, typecheck, then test
lint: format typecheck test

format:
    uv run ruff check --fix
    uv run ruff format .

test:
    uv run {{test_extras}} {{UV_RUN_ARGS}} pytest {{TEST_PATHS}} -v {{PYTEST_ARGS}}

typecheck:
    uv run {{test_extras}} ty check

check-migrations:
    uv run {{test_extras}} django-admin makemigrations --check --dry-run --settings=tests.settings --pythonpath=.

# Tag a release, for example: just tag v0.1.0rc1
tag version:
    git tag -a {{version}} -m "Release {{version}}"

build:
    uv build

publish:
    uv publish

release version: (tag version) build publish

docs-graphs:
    uv run --group docs python docs/graph.py

docs-build: docs-graphs
    uv run --group docs dokk check --root docs
    uv run --group docs dokk build --root docs

docs-serve:
    uv run --group docs dokk serve --root docs
