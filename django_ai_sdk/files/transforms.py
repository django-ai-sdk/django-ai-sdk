from __future__ import annotations

import csv
import io
import json
from typing import Any, ClassVar


class BaseTransform:
    step: ClassVar[str | None] = None

    async def run(self, data: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class CSVTransform(BaseTransform):
    async def run(self, data: Any, **kwargs: Any) -> list[dict] | Any:
        if not isinstance(data, str):
            return data
        reader = csv.DictReader(io.StringIO(data))
        return list(reader)


class JSONTransform(BaseTransform):
    async def run(self, data: Any, **kwargs: Any) -> dict | list | Any:
        if not isinstance(data, str):
            return data
        return json.loads(data)


class TextTransform(BaseTransform):
    async def run(self, data: Any, **kwargs: Any) -> dict:
        if isinstance(data, str):
            return {"data": data}
        if isinstance(data, (dict, list)):
            return {"data": json.dumps(data, default=str, ensure_ascii=False)}
        return {"data": str(data)}
