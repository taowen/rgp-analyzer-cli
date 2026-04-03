from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .models import CaptureSession, DecodeOptions, DispatchIsaOptions
from .native_decode import default_decoder_lib_dir, repo_root as v2_repo_root, resolve_helper_path
from .services import decode_payload, dispatch_isa_payload, load_capture, resource_payload, stitch_payload


def _bak_root() -> Path:
    return Path(__file__).resolve().parents[2] / "bak"


def _load_v1_modules():
    bak_root = _bak_root()
    if str(bak_root) not in sys.path:
        sys.path.insert(0, str(bak_root))
    from rgp_analyzer_cli_v1.decode_bridge import build_stitch_model as v1_build_stitch_model
    from rgp_analyzer_cli_v1.native_decode import (
        default_decoder_lib_dir as v1_default_decoder_lib_dir,
        repo_root as v1_repo_root,
        resolve_helper_path as v1_resolve_helper_path,
        run_decode_helper as v1_run_decode_helper,
    )
    from rgp_analyzer_cli_v1.parser import parse_rgp as v1_parse_rgp
    from rgp_analyzer_cli_v1.resource_metadata import extract_resource_metadata as v1_extract_resource_metadata
    from rgp_analyzer_cli_v1.tinygrad_support.isa_map import map_dispatch_spans_to_isa as v1_map_dispatch_spans_to_isa

    return {
        "parse_rgp": v1_parse_rgp,
        "build_stitch_model": v1_build_stitch_model,
        "extract_resource_metadata": v1_extract_resource_metadata,
        "run_decode_helper": v1_run_decode_helper,
        "resolve_helper_path": v1_resolve_helper_path,
        "repo_root": v1_repo_root,
        "default_decoder_lib_dir": v1_default_decoder_lib_dir,
        "map_dispatch_spans_to_isa": v1_map_dispatch_spans_to_isa,
    }


def _normalize_decode(payload: dict[str, Any]) -> dict[str, Any]:
    stitched = payload.get("stitched") or {}
    overview = stitched.get("dispatch_isa_overview") or {}
    streams = stitched.get("streams") or []
    top = (((streams[0].get("annotated_hotspots") or [None])[0]) or {}) if streams else {}
    summary = top.get("stitch_summary") or {}
    return {
        "status": payload.get("status"),
        "code_object_count": payload.get("code_object_count"),
        "code_object_load_failures": payload.get("code_object_load_failures"),
        "stream_count": payload.get("stream_count"),
        "dispatch_isa_mapped": overview.get("mapped_dispatch_count"),
        "dispatch_isa_total": overview.get("dispatch_count"),
        "primary_code_object_index": summary.get("primary_code_object_index"),
        "primary_dispatch_assignment_share": summary.get("primary_dispatch_assignment_share"),
        "primary_dispatch_isa_mapped_dispatch_share": summary.get("primary_dispatch_isa_mapped_dispatch_share"),
    }


def _normalize_resources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": item.get("index"),
            "entry_point": item.get("entry_point"),
            "vgpr_count": item.get("vgpr_count"),
            "sgpr_count": item.get("sgpr_count"),
            "lds_size": item.get("lds_size"),
            "scratch_memory_size": item.get("scratch_memory_size"),
        }
        for item in items
    ]


def _normalize_dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload.get("overall_pc_summary") or []
    top = overall[0] if overall else {}
    return {
        "stream_index": payload.get("stream_index"),
        "dispatch_count": payload.get("dispatch_count"),
        "top_code_object_index": top.get("code_object_index"),
        "top_pc": top.get("pc"),
        "top_mnemonic": top.get("mnemonic"),
    }


def _normalize_stitch(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolved_entries": payload.get("resolved_entry_count"),
        "dispatch_markers": payload.get("dispatch_marker_count"),
        "bind_markers": payload.get("bind_marker_count"),
        "dispatch_api_spans": payload.get("dispatch_api_span_count"),
        "dispatch_assignments": payload.get("dispatch_assignment_count"),
        "command_buffer_spans": payload.get("command_buffer_span_count"),
        "barrier_markers": payload.get("barrier_marker_count"),
    }


def _v1_payloads(path: Path) -> dict[str, Any]:
    v1 = _load_v1_modules()
    report = v1["parse_rgp"](path)
    stitch = v1["build_stitch_model"](report)
    resource = v1["extract_resource_metadata"](report, limit=10)
    helper = v1["resolve_helper_path"](v1["repo_root"](), None, False)
    decode_result = v1["run_decode_helper"](
        report,
        helper=helper,
        decoder_lib_dir=v1["default_decoder_lib_dir"](),
        stream_limit=1,
        hotspot_limit=4,
        strict=False,
        as_json=True,
        keep_temp=False,
    )
    dispatch = v1["map_dispatch_spans_to_isa"](report, stream_index=0, dispatch_limit=8)
    return {
        "stitch": stitch,
        "resource": resource,
        "decode": decode_result["json"] or {},
        "dispatch": dispatch,
    }


def _v2_payloads(session: CaptureSession) -> dict[str, Any]:
    helper = resolve_helper_path(v2_repo_root(), None, False)
    return {
        "stitch": stitch_payload(session),
        "resource": resource_payload(session, limit=10),
        "decode": decode_payload(
            session,
            DecodeOptions(
                stream_limit=1,
                hotspot_limit=4,
                build_helper=False,
                helper=helper,
                decoder_lib_dir=default_decoder_lib_dir(),
            ),
        ),
        "dispatch": dispatch_isa_payload(session, DispatchIsaOptions(stream_index=0, dispatch_limit=8)),
    }


def compare_v1_v2(path: Path) -> dict[str, Any]:
    session = load_capture(path)
    v1 = _v1_payloads(path)
    v2 = _v2_payloads(session)

    checks = {
        "stitch_report": _normalize_stitch(v1["stitch"]) == _normalize_stitch(v2["stitch"]),
        "resource_summary": _normalize_resources(v1["resource"]) == _normalize_resources(v2["resource"]),
        "decode_sqtt": _normalize_decode(v1["decode"]) == _normalize_decode(v2["decode"]),
        "dispatch_isa_map": _normalize_dispatch(v1["dispatch"]) == _normalize_dispatch(v2["dispatch"]),
    }
    return {
        "capture": str(path),
        "checks": checks,
        "all_match": all(checks.values()),
        "v1": {
            "stitch": _normalize_stitch(v1["stitch"]),
            "resource": _normalize_resources(v1["resource"]),
            "decode": _normalize_decode(v1["decode"]),
            "dispatch": _normalize_dispatch(v1["dispatch"]),
        },
        "v2": {
            "stitch": _normalize_stitch(v2["stitch"]),
            "resource": _normalize_resources(v2["resource"]),
            "decode": _normalize_decode(v2["decode"]),
            "dispatch": _normalize_dispatch(v2["dispatch"]),
        },
    }
