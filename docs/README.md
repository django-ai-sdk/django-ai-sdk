# Django AI SDK - Documentation

The documentation site for Django AI SDK, built with [Hugo](https://gohugo.io/) and the [Hextra](https://github.com/imfing/hextra) theme. It deploys alongside the handcrafted marketing site in `public/index.html`: the landing page lives at the GitHub Pages root (`https://django-ai-sdk.github.io/django-ai-sdk/`) and the docs at the `/docs/` subpath (`https://django-ai-sdk.github.io/django-ai-sdk/docs/`).

## Prerequisites

- [Hugo (extended)](https://gohugo.io/getting-started/installing/)
- [Go](https://golang.org/doc/install)
- [uv](https://docs.astral.sh/uv/) (for the diagram generator)

## Local Development

Regenerate the architecture diagrams and start a dev server:

```shell
make docs-serve
```

Or step by step:

```shell
cd docs
hugo mod tidy
make docs-serve
```

`make docs-serve` builds the docs (and regenerates the diagrams) into `public/docs/`, then serves the whole `public/` directory, so you can browse the landing page at `http://localhost:1313/` and the docs at `http://localhost:1313/docs/` exactly as they will appear in production.

The docs use `canonifyURLs: true` with `baseURL: /docs/` (in `docs/hugo.yaml`), so all internal links in content are written as root-absolute paths like `/quickstart` or `/images/graphs/...`. Hugo resolves these against the baseURL: locally they become `/docs/quickstart`, in production `/django-ai-sdk/docs/quickstart`. The GitHub Actions workflow overrides `--baseURL /django-ai-sdk/docs/` for production. Do not switch back to `relativeURLs: true` — it produces broken links for Hextra at a subpath.

## Building for Production

```shell
make docs-build
```

This regenerates diagrams (`docs/graph.py`) and renders the site to `public/docs/` (next to the tracked `public/index.html` landing page). The GitHub Pages workflow uploads the whole `public/` directory: the landing page at the site root and the docs at `/docs/`.

## Diagrams

Architecture diagrams live in `docs/graph.py` and render to `docs/static/images/graphs/`:

```shell
make docs-graphs
```

Requires Graphviz (`dot`) installed. If a diagram is missing, `make docs-build` will fail fast and tell you.

## Update the Theme

```shell
hugo mod get -u
hugo mod tidy
```
