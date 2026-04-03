from __future__ import annotations

from typing import Any

from .legacy_render import render_inspect as render_v1_inspect
from .legacy_render import render_stitch_report as render_v1_stitch_report
from .sqtt_stitch import render_stitched_hotspots
from .tinygrad_support.isa_map import render_dispatch_isa_map


def render_inspect(session_report: dict[str, Any], *, limit: int = 10) -> str:
    return render_v1_inspect(session_report, limit=limit)


def render_stitch_report(session_report: dict[str, Any], *, limit: int = 10) -> str:
    return render_v1_stitch_report(session_report, limit=limit)


def render_decode_payload(payload: dict[str, Any]) -> str:
    lines = ["decode_sqtt_v2:"]
    lines.append(f"  status: {payload.get('status')}")
    lines.append(f"  code_object_count: {payload.get('code_object_count')}")
    lines.append(f"  code_object_load_failures: {payload.get('code_object_load_failures')}")
    lines.append(f"  stream_count: {payload.get('stream_count')}")
    for warning in payload.get("warnings", []):
        lines.append(f"  warning: {warning}")
    stitched = payload.get("stitched")
    if stitched:
        lines.append(render_stitched_hotspots(stitched))
    return "\n".join(lines)


def render_resource_payload(items: list[dict[str, Any]]) -> str:
    lines = ["resource_summary_v2:"]
    for item in items:
        lines.append(
            f"  code_object[{item['index']}] entry_point={item.get('entry_point')} "
            f"vgpr={item.get('vgpr_count')} sgpr={item.get('sgpr_count')} "
            f"lds={item.get('lds_size')} scratch={item.get('scratch_memory_size')}"
        )
    return "\n".join(lines)


def render_triage_payload(payload: dict[str, Any]) -> str:
    lines = ["shader_triage_v2:"]
    for finding in payload.get("findings", []):
        lines.append(f"  - {finding}")
    return "\n".join(lines)


def render_dispatch_payload(payload: dict[str, Any], *, limit: int = 16) -> str:
    return render_dispatch_isa_map(payload, limit=limit)
