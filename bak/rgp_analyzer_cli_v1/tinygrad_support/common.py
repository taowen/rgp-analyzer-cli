from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def default_tinygrad_path() -> Path | None:
    candidates = [
        Path.home() / "projects" / "tinygrad",
        Path("/home/taowen/projects/tinygrad"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@contextmanager
def tinygrad_import_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def resolve_tinygrad_path(path: Path | None) -> Path:
    resolved = path or default_tinygrad_path()
    if resolved is None or not resolved.exists():
        raise RuntimeError("tinygrad path not found; pass --tinygrad-path")
    return resolved
