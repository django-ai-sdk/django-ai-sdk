---
title: Memories
sidebar:
  order: 119
---

A memory is a named knowledge base the SDK indexes for retrieval. Users upload documents and files into a memory, and link the memory to threads so its contents reach the agent. This page covers the data model, the `MemoryService` API, and the ready-made memory endpoints.

## Concepts

| Model | Role |
| --- | --- |
| `Memory` | A knowledge base: name, slug, description, `is_public` / `is_hidden`. |
| `Entry` | A piece of knowledge, with text content and JSON `data`. RAG indexes entries. A file's extracted text is stored here. |
| `EntryDocument` | A file-backed upload in a memory, with a processing status lifecycle. |
| `MemoryUser` / `MemoryGroup` | Access grants on a memory with a `can_manage` flag. The three tiers are manage, read-write and public read-only. |
| `ThreadMemory` | A memory linked to a thread, with an `active` toggle. |

All `MemoryService` methods are async and check permissions in `PermissionDomain.MEMORY`. Sync code can use the sync-prefixed aliases.

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

All list methods paginate with `limit` and `offset`. Pass `limit=None` to return everything.

`create_memory` grants the creating user `can_manage=True`.

## Access for users and groups

```python
await MemoryService.list_memory_users(memory_id, user=request.user)
await MemoryService.add_memory_user(memory_id, user_id, can_manage=True, user=request.user)
await MemoryService.update_memory_user(memory_id, user_id, can_manage=True, user=request.user)
await MemoryService.remove_memory_user(memory_id, user_id, user=request.user)

await MemoryService.list_memory_groups(memory_id, user=request.user)
await MemoryService.add_memory_group(memory_id, group_id, can_manage=False, user=request.user)
await MemoryService.remove_memory_group(memory_id, group_id, user=request.user)
```

The domain's `MemoryDefaultPermission`, or your `AI_SDK_PERMISSIONS["memory"]` override, enforces access. See [Permissions](permissions.md).

## Documents

The SDK deduplicates file uploads by content hash and processes them in the background. See [Files](files.md) for the processing lifecycle.

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

## Linking memories to threads

The thread's agent retrieves from its linked memories with RAG. `AgentService.get_agent()` uses `get_thread_memories()` to decide which memories reach the model.

```python
await MemoryService.link_memory_to_thread(memory_id, thread_id, user=request.user)
await MemoryService.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
await MemoryService.list_thread_memories(thread_id, user=request.user)        # -> list[ThreadMemoryOut]
await MemoryService.get_thread_memories(thread_id, user=request.user)         # -> active, readable Memory objects

await MemoryService.bulk_connect_memories(thread_id, ["mem-1", "mem-2"], user=request.user)
await MemoryService.toggle_memory_active(thread_id, memory_id, active=False, user=request.user)
await MemoryService.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
```

`link_memories(agent_id, thread_id)` and `unlink_memories(agent_id, thread_id)` resolve the agent's configured default memories before linking. The [Views and Routing guide](../serving/views-and-routing.md#threads-and-messages) uses them.

## Memory endpoints

The contrib layers ship ready-made memory endpoints: `get_memories_router()` for django-ninja, mounted at `/memories`, and `MemoryViewSet` for DRF. See [Views and Routing](../serving/views-and-routing.md#pick-a-starting-point). List and detail responses include an `ObjectPermissions` block with `can_read`, `can_write` and `can_manage`, computed by `django_ai_sdk.views.permissions.memory_permissions()`.

:::details[Show all memory endpoints]
| Endpoint | Operation |
| --- | --- |
| `GET /memories/settings/` | Upload constraints (`max_upload_size`, `allowed_mime_types`) |
| `POST /memories/` · `GET /memories/` | Create · list memories |
| `GET /memories/{id}/` · `PUT` · `DELETE` | Get · update · delete memory |
| `POST /memories/{id}/documents/` | Upload document (202 + `DocumentUploadResponse`) |
| `GET /memories/{id}/documents/` | List documents |
| `GET /memories/{id}/documents/{doc_id}/` · `DELETE` | Get · delete document |
| `GET /memories/{id}/documents/{doc_id}/status/` | Processing status (`DocumentStatusOut`) |
| `POST /memories/{id}/documents/{doc_id}/cancel/` | Cancel a pending or processing document |
| `GET /memories/tasks/{task_id}/status/` | Processing status by task id |
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
| `GET /memories/source/document/?entry_id=&quote=` | The document behind a cited source (its `documentId`), and the page of the quotes |
| `GET /memories/source/document/{doc_id}/file/` | The original file: a PDF inline in the browser's PDF viewer, anything else downloads |
| `GET /memories/source/document/{doc_id}/page/{page}/?quote=` | That page as a PNG, the quotes marked |
:::

### Cited document pages

A client can show a cited source as the real page. Every supported file goes the same way. The SDK converts it to a PDF, finds the page with the quotes, and renders that page with the quotes marked. The conversion only serves to find and show the page. `file/` always serves the original file.

| Files | Converted with |
| --- | --- |
| `pdf` | As is |
| `docx`, `pptx`, `xlsx` | pdf-oxide's `OfficeConverter` |
| `csv` | A one-sheet workbook, then as `xlsx`, so it shows as a table |
| `md` · `txt`, `json` | `Pdf.from_markdown` · `Pdf.from_text` |

:::details[Images, other files and known limits]
Images (`png`, `jpg`, `jpeg`, `webp`, `gif`) have no text to mark, so the SDK doesn't convert them. `source/document` returns them with `page: null`. `file/` shows the image itself, inline and sandboxed like a thread file. `page/` returns a 404. Other files get a 404 on all three endpoints. A scanned PDF without a text layer renders without marks. Word tables get an extra "Col 1, Col 2" header row, and Markdown tables show as plain text.
:::

Pass the citation's passages as repeated `quote` parameters. At most 8 count, and each is cut to 400 characters.

- `source/document` finds the page by comparing letters only, so line breaks and hyphenation don't matter. A sentence repeated on many pages doesn't decide the page alone. The page that most quotes share wins, and the first one wins a tie. It returns `{id, name, url, page, page_url}`, where `name` is the original file name. For a PDF, `page_url` is `url#page=N`, a link that opens the file on that page. For other files it is `null`, because only a PDF viewer knows `#page=`. Link to `url` instead, which downloads the original.
- `source/document/{doc_id}/page/{page}` returns a PNG with the passages marked in yellow. Its `X-Mark` header gives the marked area as `top,left,bottom,right` fractions of the page, or `""` when nothing is marked, so a client can show that part of the page. Expose the header in CORS (`CORS_EXPOSE_HEADERS = ["X-Mark"]`) when the frontend runs on another origin.

All three check `VIEW_DOCUMENT` on the document's memory. They need the `files` extra :badge[optional extra]{variant="secondary"} (`pdf-oxide`, `Pillow`, `openpyxl`). The SDK converts the file on every request, which takes about 0.1 s. For your own views, the helpers work on bytes, not memories. They are `django_ai_sdk.files.previews` (`to_pdf`, `quote_page`, `render_page`, `passage_areas`) and `django_ai_sdk.views.files` (`asource_file_response`, `asource_page_response`).

:::caution
Calling pdf-oxide yourself? Call `pdf_oxide.disable_logging()` first, or use `files.previews`, which does. With its logging on, converting an Office file deadlocks.
:::
