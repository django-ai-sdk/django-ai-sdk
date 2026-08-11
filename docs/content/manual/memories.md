---
title: Memories
type: docs
weight: 119
---

A **memory** is a named knowledge base the SDK indexes for retrieval. Documents and files are uploaded into a memory, and a memory can be linked to threads so its contents reach the agent. This page covers the data model, the `MemoryService` API, and the demo's memory endpoints.

## Concepts

| Model | Role |
| --- | --- |
| `Memory` | A knowledge base: name, slug, description, `is_public` / `is_hidden`. |
| `Entry` | A piece of knowledge (text content + JSON `data`). RAG indexes entries; a file's extraction is stored here. |
| `EntryDocument` | A file-backed upload in a memory, with a processing status lifecycle. |
| `MemoryUser` / `MemoryGroup` | Access grants on a memory with a `can_manage` flag (three tiers: manage / read-write / public read-only). |
| `ThreadMemory` | A memory linked to a thread, with an `active` toggle. |

All `MemoryService` methods are async and permission-checked (`PermissionDomain.MEMORY`); sync contexts can use the sync-prefixed aliases.

## Memory CRUD

```python
from django_ai_sdk.memories.services import MemoryService

memory = await MemoryService.create_memory(
    "Company Wiki", description="Internal docs", user=request.user
)                                   # -> MemoryOut
await MemoryService.list_memories(user=request.user, limit=100, offset=0)
await MemoryService.get_memory(memory_id, user=request.user)
await MemoryService.update_memory(memory_id, name="...", description="...", is_public=True, user=request.user)
await MemoryService.delete_memory(memory_id, user=request.user)
```

`create_memory` grants the creating user `can_manage=True` automatically.

## Access (Users and Groups)

```python
await MemoryService.list_memory_users(memory_id, user=request.user)
await MemoryService.add_memory_user(memory_id, user_id, can_manage=True, user=request.user)
await MemoryService.update_memory_user(memory_id, user_id, can_manage=True, user=request.user)
await MemoryService.remove_memory_user(memory_id, user_id, user=request.user)

await MemoryService.list_memory_groups(memory_id, user=request.user)
await MemoryService.add_memory_group(memory_id, group_id, can_manage=False, user=request.user)
await MemoryService.remove_memory_group(memory_id, group_id, user=request.user)
```

Access is enforced by the domain's `MemoryDefaultPermission` (or your `AI_SDK_PERMISSIONS["memory"]` override): see [Permissions](/docs/manual/permissions/).

## Documents

File uploads are deduplicated by content hash and processed in the background. See [Files](/docs/manual/files/) for the processing lifecycle.

```python
resp = await MemoryService.upload_document(memory_id, file, user=request.user)
# -> DocumentUploadResponse(id, status="processing", task_id)

await MemoryService.list_documents(memory_id, user=request.user)              # all statuses
await MemoryService.get_document(memory_id, doc_id, user=request.user)
await MemoryService.get_document_status(doc_id, user=request.user)            # by doc id
await MemoryService.get_task_status(task_id, user=request.user)               # by task id
await MemoryService.retry_document(doc_id, user=request.user)
await MemoryService.delete_document(memory_id, doc_id, user=request.user)
```

## Linking Memories to Threads

A linked memory is one the thread's agent can retrieve from (RAG): `AgentService.get_agent()` uses `get_thread_memories()` to decide which memories reach the model.

```python
await MemoryService.link_memory_to_thread(memory_id, thread_id, user=request.user)
await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
await MemoryService.list_thread_memories(thread_id, user=request.user)        # -> list[ThreadMemoryOut]
await MemoryService.get_thread_memories(thread_id, user=request.user)         # -> active, readable Memory objects

await MemoryService.bulk_connect_memories(thread_id, ["mem-1", "mem-2"], user=request.user)
await MemoryService.toggle_memory_active(thread_id, memory_id, active=False, user=request.user)
await MemoryService.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
```

`link_memories(agent_id, thread_id)` / `unlink_memories(agent_id, thread_id)` (used in the [Views and Routing guide](/docs/views-and-routing/#threads-and-messages)) resolve the agent's configured default memories before linking.

## Demo Endpoints

The demo (`demo/apps/memories/views/`) implements a complete Ninja router (mounted at `/memories`) and a matching [experimental DRF router](/docs/views-and-routing/#ninja-or-drf). Responses include an `ObjectPermissions` block (`can_read` / `can_write` / `can_manage`) computed via `agent_permissions()`.

| Endpoint | Operation |
| --- | --- |
| `GET /memories/settings/` | Upload constraints (`max_upload_size`, `allowed_mime_types`) |
| `POST /memories/` · `GET /memories/` | Create · list memories |
| `GET /memories/{id}/` · `PUT` · `DELETE` | Get · update · delete memory |
| `POST /memories/{id}/documents/` | Upload document (202 + `DocumentUploadResponse`) |
| `GET /memories/{id}/documents/` | List documents |
| `GET /memories/{id}/documents/{doc_id}/` · `DELETE` | Get · delete document |
| `GET /memories/{id}/documents/{doc_id}/status/` | Processing status (`DocumentStatusOut`) |
| `POST /memories/{id}/link/{thread_id}/` · `DELETE` | Link · unlink thread |
| `GET /memories/thread/{thread_id}/` | List thread memories |
| `POST /memories/thread/{thread_id}/bulk/` | Bulk-connect memories |
| `POST /memories/thread/{thread_id}/files/` | Upload a thread file (202) |
| `GET /memories/thread/{thread_id}/files/` · `DELETE .../files/{doc_id}/` | List · delete thread files |
| `GET /memories/thread/{thread_id}/files/{doc_id}/status/` | Thread file processing status |
| `PATCH /memories/thread/{thread_id}/{memory_id}/` | Toggle memory active |
| `DELETE /memories/thread/{thread_id}/{memory_id}/` | Disconnect from thread |
| `GET/POST /memories/{id}/users/` · `PATCH/DELETE .../users/{user_id}/` | Manage memory users |
| `GET/POST /memories/{id}/groups/` · `DELETE .../groups/{group_id}/` | Manage memory groups |
| `GET /memories/source/{entry_id}/{chunk_id}/` | RAG source content for a citation |
