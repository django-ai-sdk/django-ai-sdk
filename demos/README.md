# Demos

Django projects built on django-ai-sdk. Each folder is a standalone project with its
own settings, URLs, database and tests; CI runs every demo's tests (`make test-demos`)
so they keep working as the SDK changes.

| Demo | What it shows |
|---|---|
| [studio](studio/) | Chat agents, swarms, deep research, memories/RAG, workflows, MCP integrations, Ninja + DRF APIs |

[`base/`](base/) is not a demo: it holds the settings every demo shares (`demo_base.settings`).

## Run a demo

From the repository root:

```bash
make setup                 # installs the SDK and demo_base (both editable) plus the demo stack
cd demos/studio
# create .env with OPENAI_API_KEY / OPENAI_API_URL (and DEBUG=True for dev)
make reset                 # fresh SQLite db + demo data
make run                   # http://localhost:8000; PORT=8001 make run for a second demo
```

Every demo keeps its own `db.sqlite3`, uploads and vector stores in its folder, so
several can run side by side on different ports.

## Add a demo

```
demos/<name>/
  manage.py          DJANGO_SETTINGS_MODULE=<name>.settings
  <name>/            settings.py, urls.py, asgi.py, wsgi.py
  apps/              the demo's own Django apps
  tests/
  pytest.ini         DJANGO_SETTINGS_MODULE = <name>.settings
  Makefile           reset / run / test targets (copy studio's)
```

Its `settings.py` loads its own `.env`, then builds on the shared settings:

```python
BASE_DIR = Path(__file__).resolve().parent.parent
environ.Env.read_env(BASE_DIR / ".env")  # before the import: DEBUG etc. come from it

from demo_base.settings import *

ROOT_URLCONF = "<name>.urls"
INSTALLED_APPS = [*INSTALLED_APPS, "apps.<yours>"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
```

`make test-demos` picks up every folder with a `manage.py`. Demos import the SDK and
`demo_base` as installed packages, never via `sys.path`.

Heads-up: if your global gitignore ignores `demos/`, add new demo files with `git add -f`.
