PHONY: setup format test typecheck tag build publish release

setup:
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
	PYTHONPATH=demo uv run pytest tests -v

typecheck:
	uv run ty check

tag:
	@read -p "Tag (e.g. v0.1.0rc1): " TAG; \
	git tag -a $$TAG -m "Release $$TAG"

build:
	uv build

publish:
	uv publish

release: tag build publish
