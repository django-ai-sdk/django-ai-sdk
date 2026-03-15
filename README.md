# Django AI SDK

**Preview Version** - This is an early preview release. APIs may change without notice.

A Django SDK for building AI-powered applications with support for multiple LLM providers, RAG (Retrieval-Augmented Generation), and streaming responses.

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

## Project Status

This is a **preview version** actively under development. Core features are functional but APIs may evolve based on feedback.
