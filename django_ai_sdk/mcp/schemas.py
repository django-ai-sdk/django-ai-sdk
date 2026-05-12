from pydantic import BaseModel, ConfigDict


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class MCPServer(BaseModel):
    url: str
    token: str
