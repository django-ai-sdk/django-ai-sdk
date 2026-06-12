from __future__ import annotations

import csv
import io
import json
from typing import Any


class BaseTransform:
    async def run(self, data: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class CSVParseTransform(BaseTransform):
    """str → list[dict] via csv.DictReader. No LLM."""

    async def run(self, data: Any, **kwargs: Any) -> list[dict] | Any:
        if not isinstance(data, str):
            return data
        reader = csv.DictReader(io.StringIO(data))
        return list(reader)


class JSONParseTransform(BaseTransform):
    """str → dict | list via json.loads. No LLM."""

    async def run(self, data: Any, **kwargs: Any) -> dict | list | Any:
        if not isinstance(data, str):
            return data
        return json.loads(data)


class ToTextTransform(BaseTransform):
    """Any → str. json.dumps for dict/list, str() otherwise. No LLM."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str, ensure_ascii=False)
        return str(data)
