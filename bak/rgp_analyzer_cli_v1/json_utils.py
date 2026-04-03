from __future__ import annotations

from typing import Any


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "size": len(value),
            "preview_hex": value[:32].hex(),
        }
    return value
