PHONY: setup setup-all format test test-demos typecheck tag build publish release docs-graphs docs-build docs-serve

DEMO_EXTRAS := --extra qdrant --extra mcp --extra files --group studio
TEST_EXTRAS := $(DEMO_EXTRAS) --extra chroma --extra providers --extra ninja --extra drf

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
	uv run $(TEST_EXTRAS) pytest tests -v

test-demos:
	for d in demos/*/; do (cd $$d && make test) || exit 1; done

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
