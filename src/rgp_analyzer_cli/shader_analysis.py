from __future__ import annotations

from typing import Any

from .tinygrad_support.isa_map import _instructions_for_code_object


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
    hotspot_candidates = _focus_hotspot_candidates(decode_stream, focus_index)
    disassembly = _load_disassembly(report, focus_index, isa_tool)
    annotated_top_pcs = [_annotate_pc(item, disassembly) for item in filtered_top_pcs[:8]]

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
        "resource_deltas": _field_delta(
            baseline_resource,
            candidate_resource,
            "vgpr_count",
            "sgpr_count",
            "lds_size",
            "scratch_memory_size",
            "wavefront_size",
        ),
        "runtime_deltas": _field_delta(
            baseline_runtime,
            candidate_runtime,
            "instructions",
            "avg_stall_per_inst",
            "stall_share_of_duration",
            "occupancy_average_active",
            "occupancy_max_active",
        ),
        "runtime_category_count_deltas": _field_delta(
            baseline_runtime_categories,
            candidate_runtime_categories,
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_category_duration_deltas": _field_delta(
            _category_profile_map(baseline_category_profiles, "duration_total"),
            _category_profile_map(candidate_category_profiles, "duration_total"),
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_category_stall_deltas": _field_delta(
            _category_profile_map(baseline_category_profiles, "stall_total"),
            _category_profile_map(candidate_category_profiles, "stall_total"),
            "VALU",
            "SALU",
            "LDS",
            "VMEM",
            "SMEM",
        ),
        "runtime_proxy_deltas": _field_delta(
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
        ),
        "runtime_hotspot_ranking_delta": _runtime_hotspot_ranking_delta(
            baseline_runtime_hotspots,
            candidate_runtime_hotspots,
        ),
        "hotspot_deltas": _field_delta(
            baseline_hotspot,
            candidate_hotspot,
            "total_duration",
            "hitcount",
            "dispatch_assignment_share",
            "dispatch_isa_share",
        ),
        "hotspot_profile_deltas": _field_delta(
            baseline_runtime.get("top_hotspot_profile") or {},
            candidate_runtime.get("top_hotspot_profile") or {},
            "avg_duration_per_hit",
            "avg_stall_per_hit",
            "stall_share_of_duration",
        ),
        "dispatch_isa_deltas": _field_delta(
            baseline_dispatch,
            candidate_dispatch,
            "mapped_dispatch_count",
            "dispatch_count",
        ),
        "dispatch_isa_pc_ranking_delta": _pc_ranking_delta(
            baseline_dispatch.get("top_pcs") or [],
            candidate_dispatch.get("top_pcs") or [],
        ),
    }


def _pick(data: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(data, dict):
        return None
    return data.get(key)


def _numeric_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    delta = after - before
    payload: dict[str, Any] = {"before": before, "after": after, "delta": delta}
    if before not in (0, 0.0):
        payload["delta_ratio"] = delta / before
    return payload


def _field_delta(before: dict[str, Any] | None, after: dict[str, Any] | None, *fields: str) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for field in fields:
        delta = _numeric_delta(_pick(before, field), _pick(after, field))
        if delta is not None:
            deltas[field] = delta
    return deltas


def _pc_signature(item: dict[str, Any]) -> str:
    return (
        f"code_object[{int(item.get('code_object_index', 0) or 0)}]"
        f":0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic') or ''} {item.get('operands') or ''}".strip()
    )


def _runtime_hotspot_signature(item: dict[str, Any]) -> str:
    symbol = item.get("symbol") or {}
    if symbol.get("name"):
        return f"{symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
    return f"0x{int(item.get('address', 0) or 0):x}"


def _pc_ranking_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_pc_signature(item): item for item in before_items}
    after_map = {_pc_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = int((before_map.get(signature) or {}).get("count", 0) or 0)
        after = int((after_map.get(signature) or {}).get("count", 0) or 0)
        rows.append({"signature": signature, "before": before, "after": after, "delta": after - before})
    rows.sort(key=lambda item: (-abs(item["delta"]), -item["after"], item["signature"]))
    return rows[:6]


def _runtime_hotspot_ranking_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_runtime_hotspot_signature(item): item for item in before_items}
    after_map = {_runtime_hotspot_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = int((before_map.get(signature) or {}).get("total_duration", 0) or 0)
        after = int((after_map.get(signature) or {}).get("total_duration", 0) or 0)
        rows.append({"signature": signature, "before": before, "after": after, "delta": after - before})
    rows.sort(key=lambda item: (-abs(item["delta"]), -item["after"], item["signature"]))
    return rows[:6]


def _category_profile_map(items: list[dict[str, Any]], field: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        name = item.get("category")
        if not name:
            continue
        out[str(name)] = item.get(field)
    return out


def _load_disassembly(
    report: dict[str, Any] | None,
    focus_index: int | None,
    isa_tool: str | None,
) -> dict[int, dict[str, Any]]:
    if report is None or focus_index is None:
        return {}
    try:
        instructions = _instructions_for_code_object(report, code_object_index=int(focus_index), tool=isa_tool)
    except Exception:
        return {}
    return {
        int(item.address): {
            "text": item.text,
            "mnemonic": item.mnemonic,
            "operands": item.operands,
            "size": item.size,
            "branch_target": item.branch_target,
        }
        for item in instructions
    }


def _annotate_pc(item: dict[str, Any], disassembly: dict[int, dict[str, Any]]) -> dict[str, Any]:
    annotated = dict(item)
    pc = int(item.get("pc", 0) or 0)
    static = disassembly.get(pc)
    if static:
        annotated["text"] = static.get("text")
        annotated["size"] = static.get("size")
        annotated["branch_target"] = static.get("branch_target")
    return annotated


def _focus_hotspot_candidates(decode_stream: dict[str, Any], focus_index: int | None) -> list[dict[str, Any]]:
    if focus_index is None:
        return []
    rows: list[dict[str, Any]] = []
    for hotspot in decode_stream.get("annotated_hotspots") or []:
        for candidate in hotspot.get("stitched_candidates") or []:
            if int(candidate.get("code_object_index", -1) or -1) != int(focus_index):
                continue
            symbol = candidate.get("symbol") or {}
            rows.append(
                {
                    "address": hotspot.get("address"),
                    "hitcount": hotspot.get("hitcount"),
                    "total_duration": hotspot.get("total_duration"),
                    "total_stall": hotspot.get("total_stall"),
                    "avg_duration_per_hit": (hotspot.get("total_duration", 0) or 0) / max(int(hotspot.get("hitcount", 0) or 0), 1),
                    "avg_stall_per_hit": (hotspot.get("total_stall", 0) or 0) / max(int(hotspot.get("hitcount", 0) or 0), 1),
                    "dispatch_assignment_share": candidate.get("dispatch_assignment_share"),
                    "dispatch_isa_mapped_dispatch_share": candidate.get("dispatch_isa_mapped_dispatch_share"),
                    "match_kind": candidate.get("match_kind"),
                    "symbol": {
                        "name": symbol.get("name"),
                        "offset": symbol.get("offset"),
                    },
                    "top_pcs": [
                        {
                            "pc": item.get("pc"),
                            "mnemonic": item.get("mnemonic"),
                            "operands": item.get("operands"),
                            "category": item.get("category"),
                            "count": item.get("count"),
                        }
                        for item in (candidate.get("dispatch_isa_top_pcs") or [])[:4]
                    ],
                }
            )
    rows.sort(
        key=lambda item: (
            -int(item.get("total_duration", 0) or 0),
            -float(item.get("dispatch_assignment_share", 0.0) or 0.0),
            -int(item.get("hitcount", 0) or 0),
        )
    )
    return rows
