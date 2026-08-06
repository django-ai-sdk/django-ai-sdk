PHONY: reset run

reset:
	rm db.sqlite3
	rm -rf ../django_ai_sdk/migrations
	uv run python manage.py makemigrations django_ai_sdk
	uv run python manage.py migrate
	uv run python manage.py load_demo_memories
	uv run python manage.py load_demo_assistants

run:
	uv run python manage.py runserver


db:
	uv run sqlite_web db.sqlite3
