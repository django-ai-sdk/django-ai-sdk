from datetime import datetime
from typing import Any
from uuid import UUID

from django.http import HttpRequest
from django_ai_sdk.assistants.services import WebAssistantService
from ninja import Router, Schema
from pydantic import ConfigDict

router = Router()


class WebAssistantOut(Schema):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    base_class: str
    model: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool
    created_at: datetime
    updated_at: datetime


class WebAssistantCreateIn(Schema):
    name: str
    slug: str = ""
    base_class: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    mcp_servers: list[str] = []
    suggestion_enabled: bool = False
    title_generation: bool = True
    max_history: int | None = None
    file_upload: bool = False


class WebAssistantUpdateIn(Schema):
    name: str | None = None
    base_class: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    mcp_servers: list[str] | None = None
    suggestion_enabled: bool | None = None
    title_generation: bool | None = None
    max_history: int | None = None
    file_upload: bool | None = None
    active: bool | None = None


class WebAssistantBaseItem(Schema):
    path: str
    name: str


class WebAssistantToolItem(Schema):
    key: str
    path: str


class Error(Schema):
    message: str


@router.get(
    "/bases/",
    response={200: list[WebAssistantBaseItem]},
    operation_id="list_web_assistant_bases",
)
def list_web_assistant_bases(request: HttpRequest) -> list[WebAssistantBaseItem]:
    from django_ai_sdk.web_assistant.config import get_web_assistant_bases

    return [
        WebAssistantBaseItem(
            path=f"{cls.__module__}.{cls.__qualname__}",
            name=cls.__name__,
        )
        for cls in get_web_assistant_bases()
    ]


@router.get(
    "/tools/",
    response={200: list[WebAssistantToolItem]},
    operation_id="list_web_assistant_tools",
)
def list_web_assistant_tools(request: HttpRequest) -> list[WebAssistantToolItem]:
    from django_ai_sdk.web_assistant.config import get_tool_registry

    return [WebAssistantToolItem(key=key, path=path) for key, path in get_tool_registry().items()]


@router.get("/", response={200: list[WebAssistantOut]}, operation_id="list_web_assistants")
async def list_web_assistants(request: HttpRequest) -> Any:
    return await WebAssistantService.all()


@router.post("/", response={200: WebAssistantOut, 400: Error}, operation_id="create_web_assistant")
async def create_web_assistant(request: HttpRequest, payload: WebAssistantCreateIn) -> Any:
    try:
        return await WebAssistantService.create(
            {
                "name": payload.name,
                "slug": payload.slug,
                "base_class": payload.base_class,
                "model": payload.model,
                "system_prompt": payload.system_prompt,
                "tools": payload.tools,
                "mcp_servers": payload.mcp_servers,
                "suggestion_enabled": payload.suggestion_enabled,
                "title_generation": payload.title_generation,
                "max_history": payload.max_history,
                "file_upload": payload.file_upload,
            },
            user=request.user,
        )
    except Exception as e:
        return 400, Error(message=str(e))


@router.get(
    "/{assistant_id}/",
    response={200: WebAssistantOut, 404: Error},
    operation_id="get_web_assistant",
)
async def get_web_assistant(request: HttpRequest, assistant_id: UUID) -> Any:
    try:
        return await WebAssistantService.get(str(assistant_id))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/{assistant_id}/",
    response={200: WebAssistantOut, 404: Error, 400: Error},
    operation_id="update_web_assistant",
)
async def update_web_assistant(
    request: HttpRequest, assistant_id: UUID, payload: WebAssistantUpdateIn
) -> Any:
    try:
        from typing import cast

        from django_ai_sdk.assistants.services import WebAssistantUpdateData

        data = cast(
            "WebAssistantUpdateData",
            {k: v for k, v in payload.model_dump().items() if v is not None},
        )
        return await WebAssistantService.update(str(assistant_id), data)
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 400, Error(message=str(e))


@router.delete(
    "/{assistant_id}/",
    response={200: WebAssistantOut, 404: Error},
    operation_id="delete_web_assistant",
)
async def delete_web_assistant(request: HttpRequest, assistant_id: UUID) -> Any:
    try:
        return await WebAssistantService.delete(str(assistant_id))
    except ValueError as e:
        return 404, Error(message=str(e))
