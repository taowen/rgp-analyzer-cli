from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from .analyze import chunk_counts, flattened_events, generate_advice
from .cache import load_json_cache, store_json_cache
from .decode_bridge import build_stitch_model
from .native_decode import default_decoder_lib_dir, repo_root as decode_repo_root, resolve_helper_path, run_decode_helper
from .parser import parse_rgp
from .resource_metadata import extract_resource_metadata
from .shader_triage import shader_triage
from .shader_focus import build_shader_focus_payload, compare_shader_focus_payloads
from .shader_focus_instructions import load_disassembly
from .shader_focus_sources import build_source_hints, build_source_isa_blocks
from .tinygrad_support.isa_map import map_dispatch_spans_to_isa
from .capture_compare import compare_triage_payloads

from .models import CaptureSession, CompareOptions, DecodeOptions, DispatchIsaOptions, ShaderFocusOptions, TriageOptions


_TEMP_DECODE_DIR_RE = re.compile(r"/tmp/rgp-analyzer-decode-[^/]+")


def _sanitize_decode_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return _TEMP_DECODE_DIR_RE.sub("/tmp/rgp-analyzer-decode-<temp>", payload)
    if isinstance(payload, list):
        return [_sanitize_decode_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _sanitize_decode_payload(value) for key, value in payload.items()}
    return payload


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
    if options.use_cache:
        cached = load_json_cache(command="decode-sqtt", capture=session.path, options=options)
        if isinstance(cached, dict):
            return cached
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
    payload = _sanitize_decode_payload(result["json"] or {})
    if options.use_cache:
        store_json_cache(command="decode-sqtt", capture=session.path, options=options, payload=payload)
    return payload


def dispatch_isa_payload(session: CaptureSession, options: DispatchIsaOptions) -> dict[str, Any]:
    if options.use_cache:
        cached = load_json_cache(command="dispatch-isa-map", capture=session.path, options=options)
        if isinstance(cached, dict):
            return cached
    payload = map_dispatch_spans_to_isa(
        session.report,
        tool=options.tool,
        stream_index=options.stream_index,
        dispatch_limit=options.dispatch_limit,
        context_packets=options.context_packets,
        tail_packets=options.tail_packets,
        packet_limit=options.packet_limit,
        mapped_limit=options.mapped_limit,
    )
    if options.use_cache:
        store_json_cache(command="dispatch-isa-map", capture=session.path, options=options, payload=payload)
    return payload


def triage_payload(session: CaptureSession, options: TriageOptions) -> dict[str, Any]:
    if options.use_cache:
        cached = load_json_cache(command="shader-triage", capture=session.path, options=options)
        if isinstance(cached, dict):
            return cached
    decode_json = decode_payload(
        session,
        DecodeOptions(
            stream_limit=1,
            hotspot_limit=options.hotspot_limit,
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            use_cache=options.use_cache,
        ),
    )
    dispatch_json = dispatch_isa_payload(
        session,
        DispatchIsaOptions(
            stream_index=0,
            dispatch_limit=6,
            context_packets=64,
            tail_packets=512,
            packet_limit=5000,
            mapped_limit=24,
            tool=options.isa_tool,
            use_cache=options.use_cache,
        ),
    )
    stitch_json = stitch_payload(session)
    resource_json = resource_payload(session, limit=options.limit, tool=options.readelf_tool)
    payload = shader_triage(
        session.report,
        helper=options.helper,
        build_helper=options.build_helper,
        decoder_lib_dir=options.decoder_lib_dir,
        isa_tool=options.isa_tool,
        readelf_tool=options.readelf_tool,
        limit=options.limit,
        hotspot_limit=options.hotspot_limit,
        precomputed_decode_json=decode_json,
        precomputed_dispatch_isa_map=dispatch_json,
        precomputed_stitch_model=stitch_json,
        precomputed_resource_summary=resource_json,
    )
    if options.use_cache:
        store_json_cache(command="shader-triage", capture=session.path, options=options, payload=payload)
    return payload


def compare_capture_payload(
    baseline: CaptureSession,
    candidate: CaptureSession,
    options: CompareOptions,
) -> dict[str, Any]:
    baseline_payload = triage_payload(
        baseline,
        TriageOptions(
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            isa_tool=options.isa_tool,
            readelf_tool=options.readelf_tool,
            limit=options.limit,
            hotspot_limit=options.hotspot_limit,
            use_cache=options.use_cache,
        ),
    )
    candidate_payload = triage_payload(
        candidate,
        TriageOptions(
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            isa_tool=options.isa_tool,
            readelf_tool=options.readelf_tool,
            limit=options.limit,
            hotspot_limit=options.hotspot_limit,
            use_cache=options.use_cache,
        ),
    )
    payload = compare_triage_payloads(baseline_payload, candidate_payload)
    payload["baseline_file"] = str(baseline.path)
    payload["candidate_file"] = str(candidate.path)
    return payload


def shader_focus_payload(session: CaptureSession, options: ShaderFocusOptions) -> dict[str, Any]:
    triage = triage_payload(
        session,
        TriageOptions(
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            isa_tool=options.isa_tool,
            readelf_tool=options.readelf_tool,
            limit=options.limit,
            hotspot_limit=options.hotspot_limit,
            use_cache=options.use_cache,
        ),
    )
    resources = resource_payload(session, limit=options.limit, tool=options.readelf_tool)
    payload = build_shader_focus_payload(
        triage,
        resources,
        code_object_index=options.code_object_index,
        report=session.report,
        isa_tool=options.isa_tool,
        source_file=options.source_file,
    )
    payload["file"] = str(session.path)
    return payload


def code_object_isa_payload(
    session: CaptureSession,
    options: ShaderFocusOptions,
    *,
    limit: int = 32,
    symbol: str = "_amdgpu_cs_main",
) -> dict[str, Any]:
    focus = shader_focus_payload(session, options)
    focus_index = focus.get("focus_code_object_index")
    disassembly = load_disassembly(session.report, focus_index, options.isa_tool)
    instructions = [
        {
            "pc": pc,
            "mnemonic": item.get("mnemonic"),
            "operands": item.get("operands"),
            "text": item.get("text"),
            "branch_target": item.get("branch_target"),
        }
        for pc, item in sorted(disassembly.items(), key=lambda kv: kv[0])[:limit]
    ]
    source_hints = {}
    source_isa_blocks: list[dict[str, Any]] = []
    if options.source_file is not None:
        source_hints = build_source_hints(
            options.source_file,
            focus.get("runtime_proxies") or {},
            focus.get("runtime_hotspot_candidates") or [],
            (focus.get("dispatch_isa") or {}).get("top_pcs") or [],
        )
        source_isa_blocks = build_source_isa_blocks(options.source_file, instructions)
    return {
        "file": str(session.path),
        "focus_code_object_index": focus_index,
        "entry_point": (focus.get("resource") or {}).get("entry_point"),
        "symbol": symbol,
        "trace_quality": focus.get("trace_quality") or {},
        "profiling_constraints": focus.get("profiling_constraints") or {},
        "dispatch_isa": focus.get("dispatch_isa") or {},
        "source_hints": source_hints,
        "source_isa_blocks": source_isa_blocks,
        "top_pcs": (focus.get("dispatch_isa") or {}).get("top_pcs") or [],
        "instructions": instructions,
    }


def compare_shader_focus_payload(
    baseline: CaptureSession,
    candidate: CaptureSession,
    options: CompareOptions,
) -> dict[str, Any]:
    baseline_payload = shader_focus_payload(
        baseline,
        ShaderFocusOptions(
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            isa_tool=options.isa_tool,
            readelf_tool=options.readelf_tool,
            limit=options.limit,
            hotspot_limit=options.hotspot_limit,
            code_object_index=options.code_object_index,
            source_file=options.source_file,
            use_cache=options.use_cache,
        ),
    )
    candidate_payload = shader_focus_payload(
        candidate,
        ShaderFocusOptions(
            build_helper=options.build_helper,
            helper=options.helper,
            decoder_lib_dir=options.decoder_lib_dir,
            isa_tool=options.isa_tool,
            readelf_tool=options.readelf_tool,
            limit=options.limit,
            hotspot_limit=options.hotspot_limit,
            code_object_index=options.code_object_index,
            source_file=options.source_file,
            use_cache=options.use_cache,
        ),
    )
    payload = compare_shader_focus_payloads(baseline_payload, candidate_payload)
    payload["baseline_file"] = str(baseline.path)
    payload["candidate_file"] = str(candidate.path)
    return payload
