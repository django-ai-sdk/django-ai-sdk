PHONY: setup format test typecheck

setup:
	uv sync --all-extras
	uv run lefthook install

lint:
	uv run ruff check --fix
	uv run ruff format .

test:
	PYTHONPATH=demo uv run pytest django_ai_sdk/tests -v

typecheck:
	uv run pyright
