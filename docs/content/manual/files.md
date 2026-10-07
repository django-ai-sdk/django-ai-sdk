---
title: Files
type: docs
weight: 118
---

How uploaded files become searchable text: the processor → transforms pipeline, the upload/processing lifecycle, and the thread file memory. The public entry point is `FilePipeline`; agents declare one per supported file type via the `file_pipelines` attribute (see the [Agents guide](/agents/#files)).

## The Pipeline

A `FilePipeline` selects a **processor** (turns a file into text) and chains optional **transforms** (restructure that text) in sequence:

```python
from django_ai_sdk.files import FilePipeline
from django_ai_sdk.files.processors import CSVFileProcessor, TextFileProcessor
from django_ai_sdk.files.transforms import CSVTransform, TextTransform

FilePipeline(
    CSVFileProcessor(),
    transforms=[CSVTransform(), TextTransform()],
)
FilePipeline(TextFileProcessor())  # plain text, no transforms
```

The first pipeline whose processor accepts the file is used. `run()` returns `PipelineResult | None`: `None` means the file was rejected or produced no text.

```python
result = await pipeline.run(file)
result.content  # raw text from the processor
result.data     # parsed transform output (dict or list), used as entry metadata
```

## Processors

Processors validate the MIME type and extract text. `is_valid(file)` checks magic bytes via `puremagic`, falling back to the `AI_SDK_ALLOWED_FILES` extension→MIME map when detection fails.

| Processor | Accepts | Output |
| --- | --- | --- |
| `TextFileProcessor` | `text/plain`, `text/markdown`, `text/x-markdown` | UTF-8 string |
| `CSVFileProcessor` | `text/csv`, `text/plain` | Raw CSV string (pair with `CSVTransform`) |
| `JSONFileProcessor` | `application/json`, `text/json` | Raw JSON string (pair with `JSONTransform`) |
| `DocxFileProcessor` | `.docx` | Paragraph text (`python-docx`) |
| `PptxFileProcessor` | `.pptx` | Slide text (`python-pptx`) |
| `XlsxFileProcessor` | `.xlsx`, `.xls` | Rows joined by `\|` (`openpyxl`) |

Binary Office formats use `BaseBinaryFileProcessor`, which also requires a matching file extension because magic bytes can't distinguish OOXML variants.

## Transforms

| Transform | Purpose |
| --- | --- |
| `CSVTransform` | CSV string → `list[dict]` |
| `JSONTransform` | JSON string → `dict` / `list` |
| `TextTransform` | Any input → `{"data": ...}` dict |

`BaseTransform` has a single `async run(data, **kwargs)` hook; transforms receive `agent=` when a pipeline runs with one. Write your own by subclassing it. For LLM extraction, the demo's `DocumentExtractionTransform` (`demos/studio/apps/agents/transforms.py`) shows the pattern: run the extracted text through an agent with a `response_format`, e.g. the SDK's `DocumentExtraction` schema (summary, keywords, entities, facts, sections, events).

## Default Pipeline and FileService

Outside an agent context, `FileService.process()` applies the default pipeline and returns the extracted text:

```python
from django_ai_sdk.files.services import FileService

content = await FileService.process(uploaded_file)  # str | None
```

The default pipeline is resolved by `get_default_file_pipeline()`:

1. `AI_SDK_MEMORY_FILE_PIPELINE`: dotted path or list of paths to a zero-argument callable returning a `FilePipeline`; the first whose `accepts(file)` is true wins.
2. Otherwise `FilePipeline(TextFileProcessor())`: text files only, no transforms.

Upload limits are surfaced to the frontend by `get_upload_settings()` (`UploadSettings(max_upload_size, allowed_mime_types)`, derived from `AI_SDK_MAX_UPLOAD_SIZE`, `AI_SDK_ALLOWED_FILES`, and each configured pipeline's processor MIME types).

## Upload and Processing Lifecycle

`MemoryService.upload_document()` / `upload_thread_file()` save the file, deduplicate by SHA-256 (`compute_file_hash`, raising `ConflictError` on a duplicate), and enqueue a background `process_document_upload` task. Each upload gets an `EntryDocument` row with a processing status:

| Status | Meaning |
| --- | --- |
| `pending` | Saved, task queued |
| `processing` | Worker running; `processing_step` shows the current pipeline stage |
| `completed` | An `Entry` was created (this is what RAG indexes) |
| `failed` | `processing_error` holds the failure message |
| `cancelled` | Cancelled between pipeline steps |

```python
from django_ai_sdk.tasks import aget_task_status

status = await aget_task_status(task_id)  # from DocumentUploadResponse.task_id
```

Key behaviors:

- **Where it runs**: with the default `django_tasks` backend the pipeline runs inline in the upload request; a worker only picks it up once you configure a durable `TASKS` backend. See [Background Tasks](/manual/settings/#background-tasks).
- **Timeouts**: a run is failed after `AI_SDK_FILE_PIPELINE_TIMEOUT` seconds (default 900); `django_tasks` has no native timeout.
- **Stale processing**: a document stuck in `processing` past its deadline (timeout + 60s) is failed as orphaned when its status is polled via `get_document_status()` / `get_task_status()`.
- **Cancellation**: `cancelled_at` is checked at each `on_step` checkpoint (between processor/transforms), never mid-call.
- **Retry**: `retry_document()` re-enqueues `failed` / `pending` / `cancelled` documents (not `processing` or `completed`).

## Storage

`EntryDocument.file` uses `STORAGES["default"]` by default, same as any other Django `FileField` — public media root included.

To keep AI-uploaded documents out of public media without touching the site-wide default, define a `"django_ai_sdk"` entry in `STORAGES` — a reserved alias django-ai-sdk checks for, the same convention Django itself uses for `"default"`/`"staticfiles"`:

```python
# settings.py
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
    "django_ai_sdk": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {
            "location": BASE_DIR / "private_media",
        },
    },
}
```

```python
AI_SDK_FILE_UPLOAD_TO = "my/custom/path/"
```

## Images and vision

An uploaded image goes through `ImageCaptionProcessor`, which asks your vision agent for a description plus a transcription of any visible text. That caption makes the image searchable, and it is what an agent without vision gets to see. Agents also get the `ask_image` tool for questions the caption doesn't answer.

Both need `AI_SDK_VISION_AGENT = "myapp.agents.ImageAgent"`: your own agent answers every image question, so you choose the model, the instructions and the generator (provider, reasoning settings, ...). Without it, images are not captioned and there is no `ask_image`.

```python
# myapp/agents.py
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run
from django_ai_sdk.generators import openai_chat


class ImageAgent(Agent):
    name = "Image describer"
    model = "gpt-4o"
    hidden = True  # never listed or chattable; only used for images
    instructions = ["Describe images precisely. Transcribe text exactly as written."]
    llm = openai_chat

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm(), model=self.model)
```

The agent is called through `Agent.run()` with the image attached to a user message whose text is the question or the caption prompt. An agent whose own model can see images (`vision = True`) gets image attachments in a chat as pixels, up to `AI_SDK_MAX_INLINE_IMAGE_BYTES`; others get the caption, capped at `AI_SDK_IMAGE_CAPTION_LIMIT` characters.

To show the user the image `ask_image` looked at, map the tool to an artifact:

```python
from django_ai_sdk.memories.tools import ASK_IMAGE_ARTIFACT, ASK_IMAGE_TOOL


class MyAgent(Agent):
    tool_artifacts = {ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}
```

See [Artifacts after a tool](../../agents/#artifacts-after-a-tool).

## Thread File Memory

Thread uploads are backed by a hidden memory created on demand per thread:

```python
await MemoryService.upload_thread_file(thread_id, file, user=request.user)  # DocumentUploadResponse
await MemoryService.list_thread_files(thread_id, user=request.user)          # list[DocumentOut]
await MemoryService.delete_thread_file(thread_id, doc_id, user=request.user)
await MemoryService.get_document_status(doc_id, user=request.user)           # DocumentStatusOut
```

The demo wires this to `GET /threads/{id}/files/meta/` (`aget_thread_file_meta`) so the frontend knows how many files a thread has and which memory holds them. See [Memories](/manual/memories/) for the full API.
