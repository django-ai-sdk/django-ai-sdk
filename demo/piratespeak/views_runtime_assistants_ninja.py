from datetime import datetime
from typing import Any
from uuid import UUID

from django.http import HttpRequest
from django_ai_sdk.assistants.services import AssistantSettingsService
from ninja import Router, Schema
from pydantic import ConfigDict

router = Router()


class AssistantSettingsOut(Schema):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    assistant: str
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


class AssistantSettingsCreateIn(Schema):
    name: str
    slug: str = ""
    assistant: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    mcp_servers: list[str] = []
    suggestion_enabled: bool = False
    title_generation: bool = True
    max_history: int | None = None
    file_upload: bool = False


class AssistantSettingsUpdateIn(Schema):
    name: str | None = None
    assistant: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    mcp_servers: list[str] | None = None
    suggestion_enabled: bool | None = None
    title_generation: bool | None = None
    max_history: int | None = None
    file_upload: bool | None = None
    active: bool | None = None


class RuntimeAssistantBaseItem(Schema):
    path: str
    name: str


class RuntimeAssistantToolItem(Schema):
    key: str
    path: str


class Error(Schema):
    message: str


@router.get(
    "/bases/",
    response={200: list[RuntimeAssistantBaseItem]},
    operation_id="list_runtime_assistant_bases",
)
def list_runtime_assistant_bases(request: HttpRequest) -> list[RuntimeAssistantBaseItem]:
    from django_ai_sdk.assistants.config import get_runtime_assistant_bases

    return [
        RuntimeAssistantBaseItem(
            path=f"{cls.__module__}.{cls.__qualname__}",
            name=cls.__name__,
        )
        for cls in get_runtime_assistant_bases()
    ]


@router.get(
    "/tools/",
    response={200: list[RuntimeAssistantToolItem]},
    operation_id="list_runtime_assistant_tools",
)
def list_runtime_assistant_tools(request: HttpRequest) -> list[RuntimeAssistantToolItem]:
    from django_ai_sdk.assistants.config import get_tool_registry

    return [
        RuntimeAssistantToolItem(key=key, path=path) for key, path in get_tool_registry().items()
    ]


@router.get("/", response={200: list[AssistantSettingsOut]}, operation_id="list_assistant_settings")
async def list_assistant_settings(request: HttpRequest) -> Any:
    return await AssistantSettingsService.all()


@router.post(
    "/", response={200: AssistantSettingsOut, 400: Error}, operation_id="create_assistant_settings"
)
async def create_assistant_settings(
    request: HttpRequest, payload: AssistantSettingsCreateIn
) -> Any:
    try:
        return await AssistantSettingsService.create(
            {
                "name": payload.name,
                "slug": payload.slug,
                "assistant": payload.assistant,
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
    response={200: AssistantSettingsOut, 404: Error},
    operation_id="get_assistant_settings",
)
async def get_assistant_settings(request: HttpRequest, assistant_id: UUID) -> Any:
    try:
        return await AssistantSettingsService.get(str(assistant_id))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/{assistant_id}/",
    response={200: AssistantSettingsOut, 404: Error, 400: Error},
    operation_id="update_assistant_settings",
)
async def update_assistant_settings(
    request: HttpRequest, assistant_id: UUID, payload: AssistantSettingsUpdateIn
) -> Any:
    try:
        from typing import cast

        from django_ai_sdk.assistants.services import AssistantUpdateData

        data = cast(
            "AssistantUpdateData",
            {k: v for k, v in payload.model_dump().items() if v is not None},
        )
        return await AssistantSettingsService.update(str(assistant_id), data)
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 400, Error(message=str(e))


@router.delete(
    "/{assistant_id}/",
    response={200: AssistantSettingsOut, 404: Error},
    operation_id="delete_assistant_settings",
)
async def delete_assistant_settings(request: HttpRequest, assistant_id: UUID) -> Any:
    try:
        return await AssistantSettingsService.delete(str(assistant_id))
    except ValueError as e:
        return 404, Error(message=str(e))
