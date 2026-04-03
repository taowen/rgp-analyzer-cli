from __future__ import annotations

from pathlib import Path
from typing import Any

from .analyze import chunk_counts, flattened_events, generate_advice
from .decode_bridge import build_stitch_model
from .native_decode import default_decoder_lib_dir, repo_root as decode_repo_root, resolve_helper_path, run_decode_helper
from .parser import parse_rgp
from .resource_metadata import extract_resource_metadata
from .shader_triage import shader_triage
from .tinygrad_support.isa_map import map_dispatch_spans_to_isa

from .models import CaptureSession, DecodeOptions, DispatchIsaOptions, TriageOptions


def load_capture(path: Path) -> CaptureSession:
    return CaptureSession(path=path, report=parse_rgp(path))


def inspect_payload(session: CaptureSession) -> dict[str, Any]:
    report = session.report
    return {
        "file": str(session.path),
        "header": report["header"],
        "size_bytes": report["size_bytes"],
        "chunk_counts": chunk_counts(report),
        "event_count": len(flattened_events(report)),
        "code_object_record_count": sum(len(chunk["records"]) for chunk in report["code_object_databases"]),
        "sqtt_desc_count": len(report["sqtt_descs"]),
        "sqtt_data_chunk_count": len(report["sqtt_data_chunks"]),
        "advice": generate_advice(report),
    }


def stitch_payload(session: CaptureSession) -> dict[str, Any]:
    return build_stitch_model(session.report)


def resource_payload(session: CaptureSession, *, limit: int = 10, tool: str | None = None) -> list[dict[str, Any]]:
    return extract_resource_metadata(session.report, limit=limit, tool=tool)


def decode_payload(session: CaptureSession, options: DecodeOptions) -> dict[str, Any]:
    resolved_helper = resolve_helper_path(decode_repo_root(), options.helper, options.build_helper)
    result = run_decode_helper(
        session.report,
        helper=resolved_helper,
        decoder_lib_dir=options.decoder_lib_dir or default_decoder_lib_dir(),
        stream_limit=options.stream_limit,
        hotspot_limit=options.hotspot_limit,
        strict=options.strict,
        as_json=True,
        keep_temp=options.keep_temp,
    )
    return result["json"] or {}


def dispatch_isa_payload(session: CaptureSession, options: DispatchIsaOptions) -> dict[str, Any]:
    return map_dispatch_spans_to_isa(
        session.report,
        tool=options.tool,
        stream_index=options.stream_index,
        dispatch_limit=options.dispatch_limit,
        context_packets=options.context_packets,
        tail_packets=options.tail_packets,
        packet_limit=options.packet_limit,
        mapped_limit=options.mapped_limit,
    )


def triage_payload(session: CaptureSession, options: TriageOptions) -> dict[str, Any]:
    return shader_triage(
        session.report,
        helper=options.helper,
        build_helper=options.build_helper,
        decoder_lib_dir=options.decoder_lib_dir,
        isa_tool=options.isa_tool,
        readelf_tool=options.readelf_tool,
        limit=options.limit,
        hotspot_limit=options.hotspot_limit,
    )
