---
title: System Architecture
sidebar:
  order: 101
---

This page covers the core abstractions, the path a request takes through the system, and the design patterns behind them.

::image[Overview]{src="../images/graphs/overview_architecture.svg" caption="The core abstractions and how they connect."}

## Components

| Component | Responsibility | Page |
| --- | --- | --- |
| `Agent` | Coordinator; configures everything | [Agent](agent.md) |
| Registry + `AgentService` | Registration and stable-ID resolution | [Agent Registry](agent-registry.md) |
| `Stream` / `Run` | Haystack integration for chat and non-chat calls | [Stream and Run](stream-and-run.md) |
| Protocol handler | Wire format (Vercel, OpenAI, custom) | [Protocol Handler](protocol-handler.md) |
| Storage | Conversation persistence | [Storage](storage.md) |
| RAG | Knowledge retrieval as tools | [RAG](rag.md) |

Each component has a base class that defines its interface, and subclasses hold the implementations. The internal message type is `ChatMessage` (`django_ai_sdk.common`). It is protocol-agnostic, so any handler can consume it.

## Request data flow

A chat request passes through the agent, the `Stream` adapter, the Haystack pipeline and the protocol handler. Storage and post-processing run beside the main path.

::image[Data Flow]{src="../images/graphs/data_flow.svg" caption="A chat request from the agent to the protocol handler, with storage and post-processing beside it."}

## Design patterns

| Pattern | Idea | Where |
| --- | --- | --- |
| Protocol-agnostic messages | Internal `ChatMessage` works with any frontend protocol | [Protocol Handler](protocol-handler.md) |
| Event-driven streaming | Normalized events decouple adapters from protocols | [Stream Events](stream-events.md) |
| Single ID source | `Stream.stream()` generates the UUID once; SSE, storage, and endpoints share it | [ID Generation](id-generation.md) |
| Storage adapter registry | Fastest-first detection of which backend holds a thread | [Custom Storage Adapters](storage-registry.md) |
| RAG provider caching | Expensive index building happens once, cached per agent + memory | [RAG](rag.md) |

Next: [Agent](agent.md), the coordinator class.
