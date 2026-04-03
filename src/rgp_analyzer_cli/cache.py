from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 9


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    root = repo_root() / ".cache" / "rgp-analyzer-cli"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def capture_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_key(*, command: str, capture: Path, options: Any) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "command": command,
        "capture": capture_identity(capture),
        "options": _normalize(options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_cache(*, command: str, capture: Path, options: Any) -> dict[str, Any] | list[Any] | None:
    key = cache_key(command=command, capture=capture, options=options)
    path = cache_root() / f"{command}-{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_json_cache(*, command: str, capture: Path, options: Any, payload: Any) -> Any:
    key = cache_key(command=command, capture=capture, options=options)
    path = cache_root() / f"{command}-{key}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
