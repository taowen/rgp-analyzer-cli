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
    lines = ["decode_sqtt:"]
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
    lines = ["resource_summary:"]
    for item in items:
        lines.append(
            f"  code_object[{item['index']}] entry_point={item.get('entry_point')} "
            f"vgpr={item.get('vgpr_count')} sgpr={item.get('sgpr_count')} "
            f"lds={item.get('lds_size')} scratch={item.get('scratch_memory_size')}"
        )
    return "\n".join(lines)


def render_triage_payload(payload: dict[str, Any]) -> str:
    lines = ["shader_triage:"]
    summary = payload.get("summary") or {}
    resource = summary.get("resource") or {}
    if resource:
        lines.append(
            "  resource: "
            f"entry_point={resource.get('entry_point')} vgpr={resource.get('vgpr_count')} "
            f"sgpr={resource.get('sgpr_count')} lds={resource.get('lds_size')} "
            f"scratch={resource.get('scratch_memory_size')} wavefront={resource.get('wavefront_size')}"
        )
    runtime = summary.get("runtime") or {}
    if runtime:
        cats = runtime.get("category_counts") or {}
        lines.append(
            "  runtime: "
            f"instructions={runtime.get('instructions')} waves={runtime.get('waves')} "
            f"VALU={cats.get('VALU', 0)} SALU={cats.get('SALU', 0)} "
            f"LDS={cats.get('LDS', 0)} VMEM={cats.get('VMEM', 0)} SMEM={cats.get('SMEM', 0)}"
        )
    decoder = summary.get("decoder") or {}
    if decoder:
        lines.append(
            "  decoder: "
            f"status={decoder.get('status')} code_objects={decoder.get('code_object_count')} "
            f"load_failures={decoder.get('code_object_load_failures')} "
            f"dispatch_isa={decoder.get('dispatch_isa_mapped')}/{decoder.get('dispatch_isa_total')} "
            f"missing_codeobj_instrumentation={decoder.get('likely_missing_codeobj_instrumentation')}"
        )
    hotspot = summary.get("top_hotspot") or {}
    if hotspot:
        symbol = hotspot.get("symbol") or {}
        symbol_text = ""
        if symbol:
            symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
        lines.append(
            "  hotspot: "
            f"code_object={hotspot.get('code_object_index')} match={hotspot.get('match_kind')} "
            f"hitcount={hotspot.get('hitcount')} duration={hotspot.get('total_duration')} "
            f"candidates={hotspot.get('candidate_count')} ambiguous={hotspot.get('ambiguous')}"
            f"{symbol_text}"
        )
    dispatch_isa = summary.get("dispatch_isa") or {}
    if dispatch_isa:
        top_pc = dispatch_isa.get("top_pc") or {}
        top_pc_text = ""
        if top_pc:
            top_pc_text = (
                f" top_pc=code_object[{top_pc.get('code_object_index')}]"
                f":0x{int(top_pc.get('pc', 0) or 0):x} {top_pc.get('mnemonic')} {top_pc.get('operands')}"
            )
        lines.append(
            "  dispatch_isa: "
            f"stream={dispatch_isa.get('stream_index')} "
            f"mapped={dispatch_isa.get('mapped_dispatch_count')}/{dispatch_isa.get('dispatch_count')}"
            f"{top_pc_text}"
        )
    stitch = summary.get("stitch") or {}
    if stitch:
        lines.append(
            "  stitch: "
            f"resolved={stitch.get('resolved_entry_count')} partial={stitch.get('partially_resolved_entry_count')} "
            f"dispatch_spans={stitch.get('dispatch_api_span_count')} "
            f"assignments={stitch.get('dispatch_span_assignment_count')} "
            f"bind_markers={stitch.get('bind_marker_count')} "
            f"cb_spans={stitch.get('command_buffer_span_count')} "
            f"barriers={stitch.get('barrier_marker_count')}"
        )
    findings = payload.get("findings") or []
    if findings:
        lines.append("  observations:")
        for finding in findings[:4]:
            lines.append(f"    - {finding}")
        if len(findings) > 4:
            lines.append(f"    - ... {len(findings) - 4} more in --json")
    return "\n".join(lines)


def render_dispatch_payload(payload: dict[str, Any], *, limit: int = 16) -> str:
    return render_dispatch_isa_map(payload, limit=limit)
