PHONY: setup setup-all format test typecheck tag build publish release docs-graphs docs-build docs-serve

DEMO_EXTRAS := --extra qdrant --extra mcp --extra files --group demo
TEST_EXTRAS := $(DEMO_EXTRAS) --extra chroma --extra providers

setup:
	uv sync $(DEMO_EXTRAS)
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
	PYTHONPATH=demo uv run $(TEST_EXTRAS) pytest tests -v

typecheck:
	uv run $(TEST_EXTRAS) ty check

tag:
	@read -p "Tag (e.g. v0.1.0rc1): " TAG; \
	git tag -a $$TAG -m "Release $$TAG"

build:
	uv build

publish:
	uv publish

release: tag build publish

docs-graphs:
	uv run python docs/graph.py

docs-build: docs-graphs
	hugo --source docs --baseURL /docs/ --destination ../public/docs --gc --cleanDestinationDir

docs-serve: docs-build
	python3 -m http.server 1313 --directory public
