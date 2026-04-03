from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureSession:
    path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class DecodeOptions:
    stream_limit: int = 0
    hotspot_limit: int = 8
    build_helper: bool = False
    helper: Path | None = None
    decoder_lib_dir: Path | None = None
    strict: bool = False
    keep_temp: bool = False


@dataclass(frozen=True)
class DispatchIsaOptions:
    stream_index: int = 0
    dispatch_limit: int = 8
    context_packets: int = 64
    tail_packets: int = 512
    packet_limit: int = 20000
    mapped_limit: int = 32
    tool: str | None = None


@dataclass(frozen=True)
class TriageOptions:
    build_helper: bool = False
    helper: Path | None = None
    decoder_lib_dir: Path | None = None
    isa_tool: str | None = None
    readelf_tool: str | None = None
    limit: int = 10
    hotspot_limit: int = 8
