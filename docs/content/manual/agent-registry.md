---
title: Agent Registry
type: docs
weight: 103
---

How agents register themselves and get resolved by stable ID.

## Registration

Every concrete `Agent` subclass registers in `django_ai_sdk.agents.registry` on definition via `__init_subclass__`. The two public mechanisms are just ways to ensure the module is imported:

**Settings-based** (recommended):

```python
# settings.py
AI_SDK_AGENTS = ["myapp.agents.PirateAgent"]
```

**Decorator**:

```python
from django_ai_sdk.agents import auto_register

@auto_register
class PirateAgent(Agent):
    ...
```

Both work together. Skipped classes: abstract shared bases (`abstract = True`) and self-registering subclasses (e.g. runtime agents). A class is registered only once.

## Stable IDs

Each agent receives a **deterministic UUID5** derived from `module.ClassName`, so IDs never change between restarts or deployments:

```python
str(uuid.uuid5(NAMESPACE, "myapp.agents.PirateAgent"))
```

## Registry API

```python
from django_ai_sdk.agents.registry import registry

agent = registry.get(agent_id)          # by stable UUID
for agent_id in registry.ids():         # all registered agent IDs
    ...
```

`registry.setup()` instantiates registered classes. It is populated when agent modules are imported (via `AI_SDK_AGENTS`, `@auto_register`, or your own `ready()` hook).

## AgentService

Prefer `AgentService` over the raw registry: it also resolves runtime (DB-configured) agents:

```python
from django_ai_sdk.agents.services import AgentService

agent = await AgentService.get(agent_id)
```

Next: [Stream and Run](../stream-and-run/), the adapter hooks' return values.
