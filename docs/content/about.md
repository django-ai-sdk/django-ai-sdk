---
title: About
type: about
---

# Why Django AI SDK Exists

## The Story

It started with a simple frustration.

We had a Django project. We wanted to add AI. And suddenly we were drowning in complexity: managing streaming connections, handling conversation state, choosing between a dozen different SDKs, each with their own patterns and lock-in.

**It felt wrong.**

Django has always been about making the hard things simple. You don't configure databases for hours—you run `migrate` and start building. You don't wrestle with request handling—you write a view function. The framework gets out of your way.

AI development needed that same philosophy.

So we built Django AI SDK—not as a wrapper around another framework, but as a natural extension of Django itself. The same patterns you know. The same simplicity you love. But now, with AI superpowers.

---

## Our Philosophy

### **1. Convention Over Configuration**

In Django, you don't spend your first day configuring. You run `startproject` and you're building features. We believe AI should be the same.

Set a name. Write some instructions. That's it. The SDK handles streaming, storage, threading, and protocols. Everything else is optional.

```python
class MyAssistant(Assistant):
    name = "Helper Bot"
    instructions = "Be helpful"
    
    async def get_pipeline_adapter(self, thread_id=None):
        return OpenAIAdapter(client=AsyncOpenAI())
```

Three lines. Production-ready AI endpoint.

### **2. Django-Native, Not Django-Compatible**

There's a difference.

*Compatible* means "it works with Django." *Native* means "it *is* Django."

We use Django's class-based patterns. Django's async views. Django's models and ORM. You don't learn a new framework—you use the one you already know.

### **3. Frontend Freedom**

Your frontend team shouldn't be forced into a specific library. We speak the standard Vercel AI SDK protocol, which means:

- React developers use `@ai-sdk/react`
- Vue developers use compatible adapters
- Mobile apps consume the same SSE streams
- Vanilla JavaScript works out of the box

One backend. Any frontend.

### **4. Zero Switch Costs**

Start with OpenAI today. Migrate to a local Llama model tomorrow. Add Haystack for complex pipelines next month. 

Change one line in your assistant. Your views stay identical. Your frontend stays identical. Your data stays intact.

The adapter pattern means you're never locked in.

### **5. Production-Ready from Day One**

We didn't build this for demos. We built it for real applications:

- **Conversation persistence** — Messages and threads stored automatically
- **Async streaming** — Non-blocking SSE with Django's ASGI support
- **Tool integration** — Your AI calls Python functions seamlessly
- **RAG support** — Connect knowledge bases without infrastructure headaches
- **Type safety** — Pydantic models catch errors before they reach production

You focus on building features. We handle the infrastructure.

---

## What Makes This Different?

### **Not a Wrapper**

Most AI SDKs for Django are thin wrappers around OpenAI or LangChain. You still learn their patterns, their configuration, their limitations.

Django AI SDK is built *for* Django, from the ground up. The Assistant pattern, the adapter system, the protocol handlers—they're Django patterns applied to AI.

### **Not a Black Box**

Every component is transparent. You can see exactly how data flows from request to response. You can extend any piece. You can replace any piece.

The [Developer Manual](/docs/manual/) documents every internal mechanism—not because you need to know it to use the SDK, but because we believe in building systems you can understand and extend.

### **Not Just for Experts**

You don't need a PhD in machine learning. You don't need to understand transformer architectures. If you can write a Django view, you can build an AI assistant.

The complexity is there when you need it (custom adapters, advanced RAG, tool chains), but it never blocks you from getting started.

---

## Built for the Long Term

We're not chasing the latest AI trend. We're building infrastructure.

- **Stable APIs** — We won't break your code every month
- **Backend Agnostic** — When GPT-5 comes out, swap one line
- **Django Aligned** — We follow Django's release cycle and philosophy
- **Community Driven** — Open source, with contributions welcome

---

## Join Us

Django AI SDK is open source because we believe the best tools are built together.

Whether you're:
- **Building your first AI feature** — [Start with the Quick Start](/docs/quickstart)
- **Scaling a production system** — [Explore the Guides](/docs)
- **Contributing to the project** — [Read the Developer Manual](/docs/manual)

You're welcome here.

---

*"The web framework for perfectionists with deadlines."*

That's Django. And now, that's Django with AI too.

— The Django AI SDK Team
