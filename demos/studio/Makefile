PHONY: reset run db test

PORT ?= 8000
UV := uv run --extra qdrant --extra mcp --extra files --group studio

reset:
	rm -f db.sqlite3
	$(UV) python manage.py migrate
	$(UV) python manage.py load_demo_memories
	$(UV) python manage.py load_demo_agents

run:
	$(UV) python manage.py runserver $(PORT)

db:
	uv run sqlite_web db.sqlite3

test:
	$(UV) pytest -v
