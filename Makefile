PHONY: setup setup-all format test typecheck check-migrations tag build publish release docs-graphs docs-build docs-serve

SETUP_EXTRAS := --extra qdrant --extra mcp --extra files --extra dbos
# Leaves out local: its onnxruntime stack is heavy, and the tests fake the model.
TEST_EXTRAS := $(SETUP_EXTRAS) --extra chroma --extra providers --extra ninja --extra drf --extra typesafe
# example: make test UV_RUN_ARGS='--with "django==5.2.*"' TEST_PATHS=tests/oversight
UV_RUN_ARGS ?=
TEST_PATHS ?= tests
PYTEST_ARGS ?=

setup:
	uv sync $(SETUP_EXTRAS)
	uv run lefthook install

setup-all:
	uv sync --all-extras
	uv run lefthook install

lint:
	make format
	make typecheck
	make test

format:
	uv run ruff check --fix
	uv run ruff format .

test:
	uv run $(TEST_EXTRAS) $(UV_RUN_ARGS) pytest $(TEST_PATHS) -v $(PYTEST_ARGS)

typecheck:
	uv run $(TEST_EXTRAS) ty check

check-migrations:
	uv run $(TEST_EXTRAS) django-admin makemigrations --check --dry-run --settings=tests.settings --pythonpath=.

tag:
	@read -p "Tag (e.g. v0.1.0rc1): " TAG; \
	git tag -a $$TAG -m "Release $$TAG"

build:
	uv build

publish:
	uv publish

release: tag build publish

docs-graphs:
	uv run --group docs python docs/graph.py

docs-build: docs-graphs
	uv run --group docs dokk check --root docs
	uv run --group docs dokk build --root docs

docs-serve:
	uv run --group docs dokk serve --root docs
