# Demos

Django projects built on django-ai-sdk. Each folder is a standalone project with its
own settings and tests; CI runs every demo's tests (`make test-demos`) so they keep
working as the SDK changes.

| Demo | What it shows |
|---|---|
| [studio](studio/) | Chat agents, swarms, deep research, memories/RAG, workflows, MCP integrations, Ninja + DRF APIs |

## Run a demo

From the repository root:

```bash
make setup                 # installs the SDK (editable) plus the demo stack
cd demos/studio
# create .env with OPENAI_API_KEY / OPENAI_API_URL (and DEBUG=True for dev)
make reset                 # fresh SQLite db + demo data
make run
```

## Add a demo

Create `demos/<name>/` with `manage.py`, a `Makefile` exposing a `test` target, and a
`pytest.ini` pointing at its settings. It is picked up by `make test-demos`
automatically. Demos import the SDK as an installed package, never via `sys.path`.
