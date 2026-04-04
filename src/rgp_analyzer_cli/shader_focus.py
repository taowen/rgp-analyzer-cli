from __future__ import annotations

from pathlib import Path
from typing import Any

from .shader_focus_instructions import (
    annotate_pc,
    build_instruction_ranking,
    focused_runtime_hotspot_deltas,
    focus_hotspot_candidates,
    instruction_ranking_delta,
    load_disassembly,
    pc_ranking_delta,
    runtime_hotspot_ranking_delta,
)
from .shader_focus_regions import build_source_region_deltas, build_source_region_metrics
from .shader_focus_runtime import (
    build_bottleneck_hint_deltas,
    build_bottleneck_hints,
    build_event_barrier_context,
    build_memory_access_hints,
    build_occupancy_detail,
    build_tuning_summary,
    build_tuning_summary_delta,
    category_profile_map,
    field_delta,
)
from .shader_focus_sources import (
    build_source_delta_hints,
    build_source_hints,
    build_source_isa_blocks,
)


def focus_code_object_index(
    summary: dict[str, Any],
    resource_items: list[dict[str, Any]],
    explicit_index: int | None = None,
) -> int | None:
    if explicit_index is not None:
        return int(explicit_index)
    hotspot = summary.get("top_hotspot") or {}
    if hotspot.get("code_object_index") is not None:
        return int(hotspot.get("code_object_index"))
    stitch = summary.get("stitch") or {}
    if stitch.get("dominant_dispatch_code_object_index") is not None:
        return int(stitch.get("dominant_dispatch_code_object_index"))
    if len(resource_items) == 1:
        return int(resource_items[0].get("index", 0) or 0)
    return None


def build_shader_focus_payload(
    triage_payload: dict[str, Any],
    resource_items: list[dict[str, Any]],
    *,
    code_object_index: int | None = None,
    report: dict[str, Any] | None = None,
    isa_tool: str | None = None,
    source_file: Path | None = None,
) -> dict[str, Any]:
    summary = triage_payload.get("summary") or {}
    focus_index = focus_code_object_index(summary, resource_items, explicit_index=code_object_index)

    focus_resource = None
    if focus_index is not None:
        for item in resource_items:
            if int(item.get("index", -1) or -1) == focus_index:
                focus_resource = item
                break
    if focus_resource is None and resource_items:
        focus_resource = resource_items[0]

    dispatch_isa = summary.get("dispatch_isa") or {}
    dispatch_isa_map = ((triage_payload.get("experimental_signals") or {}).get("dispatch_isa_map") or {})
    overall_pc_summary = dispatch_isa_map.get("overall_pc_summary") or []
    top_pcs = overall_pc_summary or dispatch_isa.get("top_pcs") or []
    if focus_index is not None:
        filtered_top_pcs = [item for item in top_pcs if int(item.get("code_object_index", -1) or -1) == focus_index]
    else:
        filtered_top_pcs = list(top_pcs)

    runtime = summary.get("runtime") or {}
    stitch = summary.get("stitch") or {}
    profiling_constraints = summary.get("profiling_constraints") or {}
    trace_quality = summary.get("trace_quality") or {}
    hotspot = summary.get("top_hotspot") or {}
    decode_stream = ((triage_payload.get("runtime_signals") or {}).get("decode_stream") or {})
    runtime_profile = decode_stream.get("runtime_profile") or {}

    runtime_hotspots = runtime.get("top_hotspot_profiles") or []
    category_profiles = runtime_profile.get("category_profiles") or []
    wave_state_profiles = runtime_profile.get("wave_state_profiles") or []
    disassembly = load_disassembly(report, focus_index, isa_tool)
    hotspot_candidates = focus_hotspot_candidates(decode_stream, focus_index, disassembly)
    annotated_top_pcs = [annotate_pc(item, disassembly) for item in filtered_top_pcs[:8]]
    source_hints = build_source_hints(source_file, runtime_profile, hotspot_candidates, annotated_top_pcs)
    source_isa_blocks = build_source_isa_blocks(
        source_file,
        [{"pc": pc, **item} for pc, item in disassembly.items()],
    )
    instruction_ranking = build_instruction_ranking(annotated_top_pcs, hotspot_candidates)
    source_region_metrics = build_source_region_metrics(
        source_isa_blocks,
        instruction_ranking,
        hotspot_candidates,
        runtime_profile,
    )
    memory_access_hints = build_memory_access_hints(runtime_profile, source_hints)
    bottleneck_hints = build_bottleneck_hints(runtime_profile, focus_resource, memory_access_hints)
    occupancy_detail = build_occupancy_detail(runtime_profile, focus_resource)
    event_barrier_context = build_event_barrier_context(stitch)
    tuning_summary = build_tuning_summary(
        bottleneck_hints=bottleneck_hints,
        event_barrier_context=event_barrier_context,
        instruction_ranking=instruction_ranking,
        source_hints=source_hints,
    )

    code_object_catalog: list[dict[str, Any]] = []
    dispatch_histogram = {}
    stitch_payload = triage_payload.get("stitch_signals") or triage_payload.get("stitch_model") or {}
    dispatch_histogram = stitch_payload.get("dispatch_assignment_histogram") or {}
    for item in resource_items:
        idx = int(item.get("index", 0) or 0)
        dispatch_count = int(dispatch_histogram.get(str(idx), dispatch_histogram.get(idx, 0)) or 0)
        code_object_catalog.append(
            {
                "index": idx,
                "entry_point": item.get("entry_point"),
                "vgpr_count": item.get("vgpr_count"),
                "sgpr_count": item.get("sgpr_count"),
                "lds_size": item.get("lds_size"),
                "scratch_memory_size": item.get("scratch_memory_size"),
                "dispatch_assignment_count": dispatch_count,
            }
        )
    code_object_catalog.sort(key=lambda item: (0 if item["index"] == focus_index else 1, -item["dispatch_assignment_count"], item["index"]))

    enough_for_shader_tuning = (
        trace_quality.get("runtime_evidence_level") == "dispatch_isa"
        and not bool(profiling_constraints.get("submit_dilution_suspected"))
        and bool(filtered_top_pcs or dispatch_isa.get("mapped_dispatch_count"))
    )

    return {
        "file": triage_payload.get("file"),
        "focus_code_object_index": focus_index,
        "enough_for_shader_tuning": enough_for_shader_tuning,
        "trace_quality": trace_quality,
        "profiling_constraints": profiling_constraints,
        "runtime_scope": "capture_global",
        "focused_runtime_available": bool(hotspot_candidates),
        "resource": focus_resource,
        "code_object_catalog": code_object_catalog,
        "runtime": runtime,
        "runtime_hotspots": runtime_hotspots[:6],
        "runtime_proxies": {
            "stalled_instruction_share": runtime_profile.get("stalled_instruction_share"),
            "avg_wave_lifetime": runtime_profile.get("avg_wave_lifetime"),
            "max_wave_lifetime": runtime_profile.get("max_wave_lifetime"),
            "memory_instruction_share": runtime_profile.get("memory_instruction_share"),
            "memory_duration_share": runtime_profile.get("memory_duration_share"),
            "memory_stall_share": runtime_profile.get("memory_stall_share"),
            "global_memory_instruction_share": runtime_profile.get("global_memory_instruction_share"),
            "global_memory_duration_share": runtime_profile.get("global_memory_duration_share"),
            "global_memory_stall_share": runtime_profile.get("global_memory_stall_share"),
            "lds_instruction_share": runtime_profile.get("lds_instruction_share"),
            "lds_duration_share": runtime_profile.get("lds_duration_share"),
            "lds_stall_share": runtime_profile.get("lds_stall_share"),
            "scalar_duration_share": runtime_profile.get("scalar_duration_share"),
            "scalar_stall_share": runtime_profile.get("scalar_stall_share"),
            "vector_duration_share": runtime_profile.get("vector_duration_share"),
            "vector_stall_share": runtime_profile.get("vector_stall_share"),
            "sync_wait_share": runtime_profile.get("sync_wait_share"),
            "sync_wait_cycles": runtime_profile.get("sync_wait_cycles"),
            "sync_wait_cycles_per_inst": runtime_profile.get("sync_wait_cycles_per_inst"),
            "immed_instruction_share": runtime_profile.get("immed_instruction_share"),
            "immed_duration_share": runtime_profile.get("immed_duration_share"),
            "immed_stall_share": runtime_profile.get("immed_stall_share"),
            "immed_stall_per_inst": runtime_profile.get("immed_stall_per_inst"),
            "lds_stall_per_inst": runtime_profile.get("lds_stall_per_inst"),
        },
        "runtime_category_profiles": [
            {
                "category": item.get("category"),
                "count": item.get("count"),
                "count_share": item.get("count_share"),
                "duration_total": item.get("duration_total"),
                "stall_total": item.get("stall_total"),
                "duration_share": item.get("duration_share"),
                "stall_share": item.get("stall_share"),
                "stall_share_of_duration": item.get("stall_share_of_duration"),
            }
            for item in category_profiles[:6]
        ],
        "wave_state_profiles": [
            {
                "state": item.get("state"),
                "duration": item.get("duration"),
                "share": item.get("share"),
            }
            for item in wave_state_profiles[:6]
        ],
        "runtime_hotspot_candidates": hotspot_candidates[:6],
        "stitch": {
            "dominant_dispatch_code_object_index": stitch.get("dominant_dispatch_code_object_index"),
            "dominant_dispatch_code_object_share": stitch.get("dominant_dispatch_code_object_share"),
            "dispatch_api_span_count": stitch.get("dispatch_api_span_count"),
            "dispatch_span_assignment_count": stitch.get("dispatch_span_assignment_count"),
        },
        "hotspot": hotspot,
        "dispatch_isa": {
            "mapped_dispatch_count": dispatch_isa.get("mapped_dispatch_count"),
            "dispatch_count": dispatch_isa.get("dispatch_count"),
            "top_pcs": annotated_top_pcs,
        },
        "instruction_ranking": instruction_ranking,
        "source_isa_blocks": source_isa_blocks,
        "source_region_metrics": source_region_metrics,
        "memory_access_hints": memory_access_hints,
        "bottleneck_hints": bottleneck_hints,
        "occupancy_detail": occupancy_detail,
        "event_barrier_context": event_barrier_context,
        "tuning_summary": tuning_summary,
        "source_hints": source_hints,
        "capture_capabilities": {
            "trace_path": "rgp_thread_trace",
            "has_dynamic_instruction_timing": True,
            "has_dynamic_instruction_count": True,
            "has_active_lane_count": False,
            "has_not_issued_reason": False,
            "has_memory_access_stride": False,
            "has_cacheline_efficiency": False,
        },
    }


def compare_shader_focus_payloads(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_runtime = baseline.get("runtime") or {}
    candidate_runtime = candidate.get("runtime") or {}
    baseline_resource = baseline.get("resource") or {}
    candidate_resource = candidate.get("resource") or {}
    baseline_hotspot = baseline.get("hotspot") or {}
    candidate_hotspot = candidate.get("hotspot") or {}
    baseline_dispatch = baseline.get("dispatch_isa") or {}
    candidate_dispatch = candidate.get("dispatch_isa") or {}
    baseline_runtime_hotspots = baseline.get("runtime_hotspots") or []
    candidate_runtime_hotspots = candidate.get("runtime_hotspots") or []
    baseline_runtime_categories = (baseline_runtime.get("category_counts") or {})
    candidate_runtime_categories = (candidate_runtime.get("category_counts") or {})
    baseline_category_profiles = baseline.get("runtime_category_profiles") or []
    candidate_category_profiles = candidate.get("runtime_category_profiles") or []
    baseline_runtime_proxies = baseline.get("runtime_proxies") or {}
    candidate_runtime_proxies = candidate.get("runtime_proxies") or {}
    baseline_focused_hotspots = baseline.get("runtime_hotspot_candidates") or []
    candidate_focused_hotspots = candidate.get("runtime_hotspot_candidates") or []

    runtime_proxy_deltas = field_delta(
        baseline_runtime_proxies,
        candidate_runtime_proxies,
        "memory_instruction_share",
        "memory_duration_share",
        "memory_stall_share",
        "global_memory_instruction_share",
        "global_memory_duration_share",
        "global_memory_stall_share",
        "lds_instruction_share",
        "lds_duration_share",
        "lds_stall_share",
        "scalar_duration_share",
        "scalar_stall_share",
        "vector_duration_share",
        "vector_stall_share",
        "sync_wait_share",
        "sync_wait_cycles",
        "sync_wait_cycles_per_inst",
        "immed_instruction_share",
        "immed_duration_share",
        "immed_stall_share",
        "immed_stall_per_inst",
        "lds_stall_per_inst",
    )

    return {
        "baseline_file": baseline.get("file"),
        "candidate_file": candidate.get("file"),
        "focus_code_object_index": {
            "baseline": baseline.get("focus_code_object_index"),
            "candidate": candidate.get("focus_code_object_index"),
        },
        "enough_for_shader_tuning": {
            "baseline": baseline.get("enough_for_shader_tuning"),
            "candidate": candidate.get("enough_for_shader_tuning"),
        },
        "trace_quality": {
            "baseline": (baseline.get("trace_quality") or {}).get("runtime_evidence_level"),
            "candidate": (candidate.get("trace_quality") or {}).get("runtime_evidence_level"),
            "baseline_submit_dilution_suspected": (baseline.get("profiling_constraints") or {}).get("submit_dilution_suspected"),
            "candidate_submit_dilution_suspected": (candidate.get("profiling_constraints") or {}).get("submit_dilution_suspected"),
        },
        "resource_deltas": field_delta(
            baseline_resource,
            candidate_resource,
            "vgpr_count",
            "sgpr_count",
            "lds_size",
            "scratch_memory_size",
            "wavefront_size",
        ),
        "runtime_deltas": field_delta(
            baseline_runtime,
            candidate_runtime,
            "instructions",
            "stalled_instruction_share",
            "avg_stall_per_inst",
            "avg_wave_lifetime",
            "max_wave_lifetime",
            "stall_share_of_duration",
            "occupancy_average_active",
            "occupancy_max_active",
        ),
        "runtime_category_count_deltas": field_delta(
            baseline_runtime_categories,
            candidate_runtime_categories,
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_category_duration_deltas": field_delta(
            category_profile_map(baseline_category_profiles, "duration_total"),
            category_profile_map(candidate_category_profiles, "duration_total"),
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_category_stall_deltas": field_delta(
            category_profile_map(baseline_category_profiles, "stall_total"),
            category_profile_map(candidate_category_profiles, "stall_total"),
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_proxy_deltas": runtime_proxy_deltas,
        "runtime_hotspot_ranking_delta": runtime_hotspot_ranking_delta(
            baseline_runtime_hotspots,
            candidate_runtime_hotspots,
        ),
        "instruction_ranking_delta": instruction_ranking_delta(
            baseline.get("instruction_ranking") or [],
            candidate.get("instruction_ranking") or [],
        ),
        "focused_runtime_hotspot_deltas": focused_runtime_hotspot_deltas(
            baseline_focused_hotspots,
            candidate_focused_hotspots,
        ),
        "source_files": {
            "baseline": (baseline.get("source_hints") or {}).get("file"),
            "candidate": (candidate.get("source_hints") or {}).get("file"),
        },
        "source_delta_hints": build_source_delta_hints(
            baseline.get("source_hints") or {},
            candidate.get("source_hints") or {},
            runtime_proxy_deltas,
        ),
        "source_region_deltas": build_source_region_deltas(
            baseline.get("source_region_metrics") or [],
            candidate.get("source_region_metrics") or [],
        ),
        "bottleneck_hint_deltas": build_bottleneck_hint_deltas(
            baseline.get("bottleneck_hints") or [],
            candidate.get("bottleneck_hints") or [],
        ),
        "occupancy_detail_deltas": field_delta(
            baseline.get("occupancy_detail") or {},
            candidate.get("occupancy_detail") or {},
            "runtime_average_active",
            "runtime_max_active",
            "avg_wave_lifetime",
            "max_wave_lifetime",
            "stalled_instruction_share",
            "vgpr_count",
            "lds_size",
            "wavefront_size",
        ),
        "event_barrier_context_deltas": field_delta(
            baseline.get("event_barrier_context") or {},
            candidate.get("event_barrier_context") or {},
            "dispatch_api_span_count",
            "dispatch_span_assignment_count",
            "bind_marker_count",
            "command_buffer_span_count",
            "barrier_marker_count",
            "barrier_span_count",
            "unmatched_barrier_begin_count",
            "dispatches_per_cb",
            "barriers_per_dispatch",
        ),
        "tuning_summary_delta": build_tuning_summary_delta(
            baseline.get("tuning_summary") or {},
            candidate.get("tuning_summary") or {},
        ),
        "hotspot_deltas": field_delta(
            baseline_hotspot,
            candidate_hotspot,
            "total_duration",
            "hitcount",
            "dispatch_assignment_share",
            "dispatch_isa_share",
        ),
        "hotspot_profile_deltas": field_delta(
            baseline_runtime.get("top_hotspot_profile") or {},
            candidate_runtime.get("top_hotspot_profile") or {},
            "avg_duration_per_hit",
            "avg_stall_per_hit",
            "stall_share_of_duration",
        ),
        "dispatch_isa_deltas": field_delta(
            baseline_dispatch,
            candidate_dispatch,
            "mapped_dispatch_count",
            "dispatch_count",
        ),
        "dispatch_isa_pc_ranking_delta": pc_ranking_delta(
            baseline_dispatch.get("top_pcs") or [],
            candidate_dispatch.get("top_pcs") or [],
        ),
    }
