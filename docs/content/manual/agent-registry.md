---
title: Agent Registry
sidebar:
  order: 103
---

How agents register and how the SDK resolves them by stable ID.

## Registration

Every concrete `Agent` subclass registers in `django_ai_sdk.agents.registry` when the class is defined, through `__init_subclass__`. The two public mechanisms only make sure the module gets imported.

::::tabs
:::tab[Settings]
**Settings.** This is the recommended mechanism.

```python
# settings.py
AI_SDK_AGENTS = ["myapp.agents.PirateAgent"]
```
:::

:::tab[Decorator]
**Decorator.**

```python
from django_ai_sdk.agents import auto_register

@auto_register
class PirateAgent(Agent):
    ...
```
:::
::::

Both mechanisms work together. The registry skips abstract shared bases (`abstract = True`) and self-registering subclasses, such as runtime agents. A class is registered only once.

## Stable IDs

Each agent gets a deterministic UUID5 derived from `module.ClassName`, so its ID stays the same across restarts and deployments:

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

`registry.setup()` instantiates registered classes. The registry fills up when agent modules are imported, through `AI_SDK_AGENTS`, `@auto_register` or your own `ready()` hook.

## AgentService

Use `AgentService` instead of the raw registry. It also resolves runtime agents, which are configured in the database:

```python
from django_ai_sdk.agents.services import AgentService

agent = await AgentService.get(agent_id)
```

Next: [Stream and Run](stream-and-run.md), the adapter hooks' return values.
