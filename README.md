# Django AI SDK

> **Pre-release** — We're building in the open. Things move fast, and that's by design.

A Django SDK for building AI-powered applications with support for multiple LLM providers, RAG (Retrieval-Augmented Generation), and streaming responses.

## Project Status — Read This First

This is an **early preview**. We're actively iterating on the API and learning from real usage. Here's what that means for you:

- **Expect breaking changes** — APIs will shift as we find better patterns.
- **Migrations will be reset** — Don't rely on database schema stability between versions.
- **Not for production** — Use this for experimentation, prototypes, and side projects. Keep critical workloads elsewhere.
- **Watch the repo** — Things change quickly. Star & watch to stay in the loop.
- **Your feedback shapes the SDK** — Break things, open issues, tell us what hurts.

We'd love to have you along for the ride — just keep your seatbelt on.

## Features

- Multi-provider LLM support
- RAG pipeline with multiple retrieval strategies
- Streaming SSE responses
- Pluggable storage (memory/database)
- Framework-agnostic core design

## Quick Start

### Setup

```bash
make setup
```

### Run Tests

```bash
make test
```

### Run Demo Application

```bash
cd demo/
make run
```

The demo application is located in the `demo/` directory and showcases a conversational AI assistant with RAG capabilities.
