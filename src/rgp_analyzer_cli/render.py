from __future__ import annotations

from typing import Any

from .legacy_render import render_inspect as render_v1_inspect
from .legacy_render import render_stitch_report as render_v1_stitch_report
from .render_focus import (
    render_code_object_isa_payload,
    render_shader_focus_payload,
)
from .render_focus_compare import render_compare_shader_focus_payload
from .render_regions import render_compare_region_focus_payload, render_region_focus_payload
from .sqtt_stitch import render_stitched_hotspots
from .tinygrad_support.isa_map import render_dispatch_isa_map


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def _fmt_ratio_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "n/a"


def _fmt_metric_row(item: dict[str, Any]) -> str:
    return (
        f"metric={item.get('name')} "
        f"value={_fmt_float(item.get('value'))} "
        f"threshold={_fmt_float(item.get('threshold'))}"
    )


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
        extras = []
        if isinstance(runtime.get("avg_stall_per_inst"), (int, float)):
            extras.append(f"avg_stall={runtime.get('avg_stall_per_inst'):.2f}")
        if isinstance(runtime.get("stall_share_of_duration"), (int, float)):
            extras.append(f"stall_share={runtime.get('stall_share_of_duration'):.2f}")
        if isinstance(runtime.get("occupancy_average_active"), (int, float)):
            extras.append(f"occ_avg={runtime.get('occupancy_average_active'):.2f}")
        if runtime.get("occupancy_max_active") is not None:
            extras.append(f"occ_max={runtime.get('occupancy_max_active')}")
        lines.append(
            "  runtime: "
            f"instructions={runtime.get('instructions')} waves={runtime.get('waves')} "
            f"VALU={cats.get('VALU', 0)} SALU={cats.get('SALU', 0)} "
            f"LDS={cats.get('LDS', 0)} VMEM={cats.get('VMEM', 0)} SMEM={cats.get('SMEM', 0)}"
            + (f" {' '.join(extras)}" if extras else "")
        )
        top_category = runtime.get("top_category") or {}
        if top_category:
            category_bits = [
                f"category={top_category.get('category')}",
                f"count={top_category.get('count')}",
                f"duration={top_category.get('duration_total')}",
                f"stall={top_category.get('stall_total')}",
            ]
            if isinstance(top_category.get("stall_share_of_duration"), (int, float)):
                category_bits.append(f"stall_share={top_category.get('stall_share_of_duration'):.2f}")
            lines.append("  runtime_top_category: " + " ".join(category_bits))
        top_wave_state = runtime.get("top_wave_state") or {}
        if top_wave_state:
            state_line = (
                f"  runtime_top_wave_state: state={top_wave_state.get('state')} "
                f"duration={top_wave_state.get('duration')}"
            )
            if isinstance(top_wave_state.get("share"), (int, float)):
                state_line += f" share={top_wave_state.get('share'):.2f}"
            lines.append(state_line)
        top_hotspot_profile = runtime.get("top_hotspot_profile") or {}
        if top_hotspot_profile:
            symbol = top_hotspot_profile.get("symbol") or {}
            symbol_text = ""
            if symbol.get("name"):
                symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
            line = (
                f"  runtime_top_hotspot: address=0x{int(top_hotspot_profile.get('address', 0) or 0):x} "
                f"duration={top_hotspot_profile.get('total_duration')} "
                f"stall={top_hotspot_profile.get('total_stall')} "
                f"hitcount={top_hotspot_profile.get('hitcount')}"
            )
            if isinstance(top_hotspot_profile.get("avg_duration_per_hit"), (int, float)):
                line += f" avg_duration={top_hotspot_profile.get('avg_duration_per_hit'):.2f}"
            if isinstance(top_hotspot_profile.get("avg_stall_per_hit"), (int, float)):
                line += f" avg_stall={top_hotspot_profile.get('avg_stall_per_hit'):.2f}"
            if isinstance(top_hotspot_profile.get("stall_share_of_duration"), (int, float)):
                line += f" stall_share={top_hotspot_profile.get('stall_share_of_duration'):.2f}"
            line += symbol_text
            lines.append(line)
        hotspot_profiles = runtime.get("top_hotspot_profiles") or []
        if len(hotspot_profiles) > 1:
            lines.append("  runtime_hotspot_ranking:")
            for item in hotspot_profiles[:3]:
                symbol = item.get("symbol") or {}
                symbol_text = ""
                if symbol.get("name"):
                    symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
                line = (
                    f"    - address=0x{int(item.get('address', 0) or 0):x} "
                    f"duration={item.get('total_duration')} stall={item.get('total_stall')} "
                    f"hitcount={item.get('hitcount')}"
                )
                if isinstance(item.get("avg_stall_per_hit"), (int, float)):
                    line += f" avg_stall={item.get('avg_stall_per_hit'):.2f}"
                if isinstance(item.get("stall_share_of_duration"), (int, float)):
                    line += f" stall_share={item.get('stall_share_of_duration'):.2f}"
                line += symbol_text
                lines.append(line)
    decoder = summary.get("decoder") or {}
    if decoder:
        lines.append(
            "  decoder: "
            f"status={decoder.get('status')} code_objects={decoder.get('code_object_count')} "
            f"load_failures={decoder.get('code_object_load_failures')} "
            f"dispatch_isa={decoder.get('dispatch_isa_mapped')}/{decoder.get('dispatch_isa_total')} "
            f"missing_codeobj_instrumentation={decoder.get('likely_missing_codeobj_instrumentation')} "
            f"sparse_runtime_trace={decoder.get('sparse_runtime_trace')}"
        )
    trace_quality = summary.get("trace_quality") or {}
    if trace_quality:
        lines.append(
            "  trace_quality: "
            f"level={trace_quality.get('runtime_evidence_level')} "
            f"sqtt_bytes={trace_quality.get('sqtt_trace_bytes')} "
            f"queue_events={trace_quality.get('queue_event_count')} "
            f"instructions={trace_quality.get('decoded_instruction_count')} "
            f"waves={trace_quality.get('decoded_wave_count')} "
            f"dispatch_spans={trace_quality.get('dispatch_span_count')} "
            f"mapped_dispatch={trace_quality.get('mapped_dispatch_count')}/{trace_quality.get('total_dispatch_count')}"
        )
    profiling_constraints = summary.get("profiling_constraints") or {}
    if profiling_constraints:
        line = (
            "  profiling_constraints: "
            f"submit_dilution_suspected={profiling_constraints.get('submit_dilution_suspected')}"
        )
        if profiling_constraints.get("sqtt_trace_bytes") is not None:
            line += f" sqtt_bytes={profiling_constraints.get('sqtt_trace_bytes')}"
        if profiling_constraints.get("dispatch_span_count") is not None:
            line += f" dispatch_spans={profiling_constraints.get('dispatch_span_count')}"
        if profiling_constraints.get("decoded_instruction_count") is not None:
            line += f" instructions={profiling_constraints.get('decoded_instruction_count')}"
        if profiling_constraints.get("sparse_runtime_trace") is not None:
            line += f" sparse_runtime_trace={profiling_constraints.get('sparse_runtime_trace')}"
        lines.append(line)
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
            f"barriers={stitch.get('barrier_marker_count')} "
            f"barrier_spans={stitch.get('barrier_span_count')} "
            f"barrier_unmatched={stitch.get('unmatched_barrier_begin_count')}"
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


def render_compare_capture_payload(payload: dict[str, Any]) -> str:
    lines = ["compare_captures:"]
    lines.append(f"  baseline: {payload.get('baseline_file')}")
    lines.append(f"  candidate: {payload.get('candidate_file')}")

    trace_quality = payload.get("trace_quality") or {}
    lines.append(
        "  trace_quality: "
        f"baseline={trace_quality.get('baseline')} "
        f"candidate={trace_quality.get('candidate')} "
        f"baseline_submit_dilution={trace_quality.get('baseline_submit_dilution_suspected')} "
        f"candidate_submit_dilution={trace_quality.get('candidate_submit_dilution_suspected')}"
    )
    for section_name in ("resource", "runtime", "decoder", "dispatch_isa", "hotspot"):
        section = payload.get(section_name) or {}
        deltas = section.get("deltas") or {}
        if deltas:
            lines.append(f"  {section_name}_deltas:")
            for key, item in deltas.items():
                line = f"    - {key}: {item.get('before')} -> {item.get('after')} (delta={item.get('delta')}"
                ratio = item.get("delta_ratio")
                if isinstance(ratio, (int, float)):
                    line += f", ratio={ratio:.3f}"
                line += ")"
                lines.append(line)

    runtime = payload.get("runtime") or {}
    runtime_hotspots = payload.get("runtime_hotspots") or []
    runtime_hotspots = payload.get("runtime_hotspots") or []
    if runtime:
        lines.append(
            "  runtime_identity: "
            f"top_category={runtime.get('baseline_top_category')} -> {runtime.get('candidate_top_category')} "
            f"top_wave_state={runtime.get('baseline_top_wave_state')} -> {runtime.get('candidate_top_wave_state')}"
        )
        category_count_deltas = runtime.get("category_count_deltas") or {}
        if category_count_deltas:
            lines.append("  runtime_category_count_deltas:")
            for key, item in category_count_deltas.items():
                lines.append(
                    f"    - {key}: {item.get('before')} -> {item.get('after')} "
                    f"(delta={item.get('delta')})"
                )
        hotspot_ranking_delta = runtime.get("hotspot_ranking_delta") or []
        if hotspot_ranking_delta:
            lines.append("  runtime_hotspot_ranking_delta:")
            for item in hotspot_ranking_delta[:4]:
                lines.append(
                    f"    - {item.get('signature')}: {item.get('before')} -> {item.get('after')} "
                    f"(delta={item.get('delta')})"
                )

    dispatch_isa = payload.get("dispatch_isa") or {}
    baseline_top_pc = dispatch_isa.get("baseline_top_pc") or {}
    candidate_top_pc = dispatch_isa.get("candidate_top_pc") or {}
    if baseline_top_pc or candidate_top_pc:
        lines.append(
            "  dispatch_isa_identity: "
            f"baseline=code_object[{baseline_top_pc.get('code_object_index')}]"
            f":0x{int(baseline_top_pc.get('pc', 0) or 0):x} {baseline_top_pc.get('mnemonic') or ''} "
            f"candidate=code_object[{candidate_top_pc.get('code_object_index')}]"
            f":0x{int(candidate_top_pc.get('pc', 0) or 0):x} {candidate_top_pc.get('mnemonic') or ''}"
        )
    pc_ranking_delta = dispatch_isa.get("pc_ranking_delta") or []
    if pc_ranking_delta:
        lines.append("  dispatch_isa_pc_ranking_delta:")
        for item in pc_ranking_delta[:4]:
            lines.append(
                f"    - {item.get('signature')}: {item.get('before')} -> {item.get('after')} "
                f"(delta={item.get('delta')})"
            )

    hotspot = payload.get("hotspot") or {}
    if hotspot:
        baseline = hotspot.get("baseline") or {}
        candidate = hotspot.get("candidate") or {}
        lines.append(
            "  hotspot_identity: "
            f"baseline={baseline.get('signature')} "
            f"candidate={candidate.get('signature')}"
        )
        profile_deltas = hotspot.get("profile_deltas") or {}
        if profile_deltas:
            lines.append("  hotspot_profile_deltas:")
            for key, item in profile_deltas.items():
                line = f"    - {key}: {item.get('before')} -> {item.get('after')} (delta={item.get('delta')}"
                ratio = item.get("delta_ratio")
                if isinstance(ratio, (int, float)):
                    line += f", ratio={ratio:.3f}"
                line += ")"
                lines.append(line)
    return "\n".join(lines)
