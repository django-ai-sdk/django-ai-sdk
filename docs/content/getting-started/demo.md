---
title: Demo project
sidebar:
  order: 2
---

[Studio](https://github.com/django-ai-sdk/studio/) is the demo project for the SDK. It lives in its own repository, so you can read a complete project without the SDK's internals around it.

- **`app/`**: a Django project with chat agents, swarms, deep research, memories and RAG, workflows with a worker, oversight, MCP integrations, and both the django-ninja and DRF APIs.
- **`mac/`**: native SwiftUI clients for macOS, iPhone and iPad. They stream chat, let you build agents and run workflows against the Django app.

## Run it

The Django app installs the SDK from a folder named `sdk` next to Studio, editable, so changes you make to the SDK show up at once. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
git clone https://github.com/django-ai-sdk/django-ai-sdk.git sdk
git clone https://github.com/django-ai-sdk/studio.git
cd studio
just app::reset    # install, create a fresh SQLite database and demo data
just app::run      # http://localhost:8000
just app::worker   # in a second terminal, for workflow runs
```

Put `OPENAI_API_KEY` and `OPENAI_API_URL` in `app/.env` first. To run it in Docker instead, use `just app::up`. `just app::up pypi` runs it against the latest release instead of your checkout.

The macOS app is on [Studio's releases page](https://github.com/django-ai-sdk/studio/releases).
