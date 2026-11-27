---
title: Files
sidebar:
  order: 118
---

This page covers how uploaded files become searchable text: the pipeline of a processor followed by transforms, the upload and processing lifecycle, and the thread file memory. The public entry point is `FilePipeline`. Agents declare one pipeline per supported file type in the `file_pipelines` attribute. See the [Agents guide](../agents/agents.md#files).

## The pipeline

A `FilePipeline` selects a **processor**, which turns a file into text. It then runs optional **transforms** in sequence, which restructure that text:

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

The SDK uses the first pipeline whose processor accepts the file. `run()` returns `PipelineResult | None`. `None` means the pipeline rejected the file or the file produced no text.

```python
result = await pipeline.run(file)
result.content  # raw text from the processor
result.data     # parsed transform output (dict or list), used as entry metadata
```

## Processors

Processors validate the MIME type and extract text. `is_valid(file)` checks magic bytes with `puremagic`. When detection fails, it falls back to the `AI_SDK_ALLOWED_FILES` map from extension to MIME type.

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

`BaseTransform` has a single `async run(data, **kwargs)` hook. Transforms receive `agent=` when a pipeline runs with an agent. Subclass `BaseTransform` to write your own. For LLM extraction, the demo's `DocumentExtractionTransform` ([`app/studio/apps/agents/transforms.py`](https://github.com/django-ai-sdk/studio/tree/main/app/studio/apps/agents/transforms.py)) shows the pattern. It runs the extracted text through an agent with a `response_format`, such as the SDK's `DocumentExtraction` schema with summary, keywords, entities, facts, sections and events.

## Default pipeline and FileService

Outside an agent context, `FileService.process()` applies the default pipeline and returns the extracted text:

```python
from django_ai_sdk.files.services import FileService

content = await FileService.process(uploaded_file)  # str | None
```

`get_default_file_pipeline()` resolves the default pipeline:

1. `AI_SDK_MEMORY_FILE_PIPELINE`, a dotted path or list of paths to a zero-argument callable that returns a `FilePipeline`. The first pipeline whose `accepts(file)` is true wins.
2. Otherwise `FilePipeline(TextFileProcessor())`, which takes text files only and has no transforms.

`get_upload_settings()` gives the upload limits to the frontend as `UploadSettings(max_upload_size, allowed_mime_types)`. It derives them from `AI_SDK_MAX_UPLOAD_SIZE`, `AI_SDK_ALLOWED_FILES` and the processor MIME types of each configured pipeline. `MemoryService.upload_document` and `upload_thread_file` reject a file over `AI_SDK_MAX_UPLOAD_SIZE` with a `UserError` (400).

## Upload and processing lifecycle

`MemoryService.upload_document()` and `upload_thread_file()` save the file and deduplicate it by SHA-256 with `compute_file_hash`, which raises `ConflictError` on a duplicate. They then enqueue a background `process_document_upload` task. Each upload gets an `EntryDocument` row with a processing status:

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

The lifecycle behaves as follows:

- **Where it runs.** With the default `django_tasks` backend, the pipeline runs inline in the upload request. A worker picks it up only once you configure a durable `TASKS` backend. See [Background Tasks](settings.md#background-tasks).
- **Timeouts.** The task fails a run after `AI_SDK_FILE_PIPELINE_TIMEOUT` seconds (default 900), because `django_tasks` has no native timeout.
- **Stale processing.** A document stuck in `processing` past its deadline, the timeout plus 60 seconds, fails as orphaned when `get_document_status()` or `get_task_status()` polls its status.
- **Cancellation.** The task checks `cancelled_at` at each `on_step` checkpoint, between the processor and the transforms, never in the middle of a call.
- **Retry.** `retry_document()` enqueues `failed`, `pending` and `cancelled` documents again, but not `processing` or `completed` ones.

## Storage

`EntryDocument.file` uses `STORAGES["default"]` by default, like any other Django `FileField`. That includes a public media root.

To keep uploaded documents out of public media without changing the site-wide default, define a `"django_ai_sdk"` entry in `STORAGES`. django-ai-sdk checks for this reserved alias, the same convention Django uses for `"default"` and `"staticfiles"`:

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

An uploaded image goes through `ImageCaptionProcessor`, which asks your vision agent for a description and a transcription of any visible text. That caption makes the image searchable, and an agent without vision sees only the caption. Agents also get the `ask_image` tool for questions the caption doesn't answer.

:::tool
```json
{
  "name": "ask_image",
  "description": "Look at an image uploaded to this thread and answer a question about it. Use this for anything about an image's visual content (what it shows, colours, text in it, details) that its description doesn't already answer.",
  "parameters": {
    "type": "object",
    "properties": {
      "image": {"type": "string", "description": "The image's document id, or its filename."},
      "question": {"type": "string", "description": "What to find out about the image, self-contained."}
    },
    "required": ["image", "question"]
  }
}
```
:::

Both need `AI_SDK_VISION_AGENT = "myapp.agents.ImageAgent"`. Your own agent answers every image question, so you choose the model, the instructions and the generator, including provider and reasoning settings.

:::note
Without the setting, the SDK does not caption images and there is no `ask_image`.
:::

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

The SDK calls the vision agent through `Agent.run()`, with the image attached to a user message. The message text is the question or the caption prompt. An agent whose own model can see images (`vision = True`) gets image attachments in a chat as pixels, up to `AI_SDK_MAX_INLINE_IMAGE_BYTES`. Other agents get the caption, capped at `AI_SDK_IMAGE_CAPTION_LIMIT` characters.

To show the user the image `ask_image` looked at, map the tool to an artifact:

```python
from django_ai_sdk.memories.tools import ASK_IMAGE_ARTIFACT, ASK_IMAGE_TOOL


class MyAgent(Agent):
    tool_artifacts = {ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}
```

See [Artifacts after a tool](../agents/agents.md#artifacts-after-a-tool).

## Thread file memory

Each thread stores its uploads in a hidden memory, created on demand:

```python
await MemoryService.upload_thread_file(thread_id, file, user=request.user)  # DocumentUploadResponse
await MemoryService.list_thread_files(thread_id, user=request.user)          # list[DocumentOut]
await MemoryService.delete_thread_file(thread_id, doc_id, user=request.user)
await MemoryService.get_document_status(doc_id, user=request.user)           # DocumentStatusOut
```

The demo connects this to `GET /threads/{id}/files/meta/` (`aget_thread_file_meta`), so the frontend knows how many files a thread has and which memory holds them. See [Memories](memories.md) for the full API.

## Fonts

pdf-oxide renders cited document pages with the system's fonts. A PDF that doesn't embed its fonts needs them, and so does every converted Word, Excel or text file.

:::caution
On a server without fonts, pdf-oxide draws every letter as a box.
:::

On Linux and in Docker, install Liberation:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
```

Liberation Sans, Serif and Mono have the same widths as Arial, Times New Roman and Courier New, so converted pages keep their spacing. pdf-oxide falls back to them by name. macOS uses its own fonts.

## Making PDFs

`markdown_to_pdf` turns Markdown into a PDF with pdf-oxide's `Pdf.from_markdown` and the system's fonts. pdf-oxide doesn't yet wrap lines, decode entities or draw tables and rules, so the SDK prepares the Markdown first:

- It wraps lines at `width` characters (80).
- It decodes `&amp;` and similar entities.
- It turns a table row into a line with its cells apart (`A1 · Eerste rij`).
- It turns a rule into an empty line.

Bold, italic and inline code show as plain text, because pdf-oxide draws them over their neighbours. Pages are Letter size. The SDK also uses `markdown_to_pdf` to preview cited Markdown sources.

```python
from django_ai_sdk.files.documents import markdown_to_pdf

pdf = markdown_to_pdf(text, title="Report")   # bytes
```
