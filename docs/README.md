# Django AI SDK - Documentation

The documentation site, built with [dokk](https://github.com/terminalkitten/dokk) from the Markdown in `content/`. It deploys alongside the landing page in `public/index.html`: the landing page at the GitHub Pages root (`https://django-ai-sdk.github.io/django-ai-sdk/`) and the docs at `/docs/` (`https://django-ai-sdk.github.io/django-ai-sdk/docs/`).

## Local development

```shell
make docs-serve    # http://127.0.0.1:8002/, reloads on save
make docs-build    # diagrams, link check, then the site into public/docs/
```

The diagrams need [Graphviz](https://graphviz.org/download/) (`brew install graphviz`). `docs/graph.py` writes them to `content/images/graphs/`.

## Writing

Pages are Markdown files under `content/`; `dokk.toml` sets the sidebar. Link to other pages by their file, relative to the current one (`[Settings](manual/settings.md)`), and use dokk's `:::note`, `::::cards` and `::::tabs` blocks for callouts, cards and tabs.
