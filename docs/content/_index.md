---
title: Django AI SDK
toc: false
---

# Django AI SDK

**Add AI capabilities to your Django project in minutes, not days.**

Django AI SDK brings the same simplicity Django is known for to AI development. No complex configuration, no framework lock-in—just clean, Pythonic patterns you already know.

```python
class MyAssistant(Assistant):
    name = "Helper Bot"
    instructions = "Be helpful and concise"
    
    async def get_pipeline_adapter(self, thread_id=None):
        return OpenAIAdapter(client=AsyncOpenAI())
```

Three lines. One endpoint. AI-powered.

{{< cards >}}
  {{< card link="docs/quickstart" title="Quick Start" icon="card" >}}
  {{< card link="docs" title="Documentation" icon="book-open" >}}
  {{< card link="about" title="Why Django AI?" icon="heart" >}}
{{< /cards >}}

## Why Developers Choose Django AI SDK

### **Django-Native by Design**

We didn't wrap an AI framework in Django—we built Django patterns for AI. Classes, async views, and models work exactly as you'd expect. No new concepts to learn.

### **Zero Frontend Lock-in**

Works with React, Vue, mobile apps, or vanilla JavaScript. We speak the standard Vercel AI SDK protocol, so your frontend team uses tools they already know.

### **Switch Costs Are Zero**

Start with OpenAI. Migrate to Haystack. Add a local Llama model. Change one line in your assistant class. Your views stay the same. Your frontend stays the same.

### **Batteries Included**

Conversation storage, message threading, streaming responses, tool calling—all production-ready from day one. You build features, not infrastructure.

---

## See It Working

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","parts":[{"type":"text","text":"Hello!"}]}]}'
```

Response (streaming):
```
data: {"type":"start","messageId":"msg_abc123"}
data: {"type":"text-delta","id":"text_001","delta":"Hello"}
data: {"type":"text-delta","id":"text_001","delta":"! How can I help?"}
data: {"type":"finish"}
data: [DONE]
```

That's Server-Sent Events (SSE) flowing straight to your frontend. No polling. No websockets. Just clean, real-time streaming.

---

## Built for Production

- **Async-first** — Non-blocking streaming with Django's ASGI support
- **Persistent conversations** — Built-in thread and message storage
- **Tool integration** — Your AI can call Python functions
- **RAG support** — Connect knowledge bases in minutes
- **Type-safe** — Pydantic models throughout

---

## Get Started

```bash
pip install django-ai-sdk openai
```

[Create your first assistant →](docs/quickstart)

---

*Django AI SDK is open source and community-driven. Join us in making AI development as simple as Django itself.*
