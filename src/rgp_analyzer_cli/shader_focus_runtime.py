from __future__ import annotations

from typing import Any


def numeric_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    delta = after - before
    payload: dict[str, Any] = {"before": before, "after": after, "delta": delta}
    if before not in (0, 0.0):
        payload["delta_ratio"] = delta / before
    return payload


def field_delta(before: dict[str, Any] | None, after: dict[str, Any] | None, *fields: str) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for field in fields:
        delta = numeric_delta(_pick(before, field), _pick(after, field))
        if delta is not None:
            deltas[field] = delta
    return deltas


def category_profile_map(items: list[dict[str, Any]], field: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        name = item.get("category")
        if not name:
            continue
        out[str(name)] = item.get(field)
    return out


def build_memory_access_hints(
    runtime_profile: dict[str, Any],
    source_hints: dict[str, Any],
) -> dict[str, Any]:
    proxies = {
        "global_memory_instruction_share": runtime_profile.get("global_memory_instruction_share"),
        "global_memory_duration_share": runtime_profile.get("global_memory_duration_share"),
        "global_memory_stall_share": runtime_profile.get("global_memory_stall_share"),
        "lds_instruction_share": runtime_profile.get("lds_instruction_share"),
        "lds_duration_share": runtime_profile.get("lds_duration_share"),
        "lds_stall_share": runtime_profile.get("lds_stall_share"),
        "lds_stall_per_inst": runtime_profile.get("lds_stall_per_inst"),
    }
    relevant_matches = [
        item
        for item in (source_hints.get("matches") or [])
        if item.get("reason") in {"global_memory", "lds_pressure"}
    ]
    return {
        "global_memory_bound_suspected": bool(
            isinstance(proxies["global_memory_duration_share"], (int, float))
            and proxies["global_memory_duration_share"] >= 0.05
        ),
        "lds_pressure_suspected": bool(
            isinstance(proxies["lds_stall_per_inst"], (int, float))
            and proxies["lds_stall_per_inst"] > 0
        ),
        "proxies": proxies,
        "source_matches": relevant_matches[:6],
    }


def build_bottleneck_hints(
    runtime_profile: dict[str, Any],
    focus_resource: dict[str, Any] | None,
    memory_access_hints: dict[str, Any],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    def add_hint(name: str, value: Any, threshold: Any) -> None:
        hints.append({"name": name, "value": value, "threshold": threshold})

    sync_wait_share = runtime_profile.get("sync_wait_share")
    immed_stall_per_inst = runtime_profile.get("immed_stall_per_inst")
    global_mem_duration_share = runtime_profile.get("global_memory_duration_share")
    lds_stall_per_inst = runtime_profile.get("lds_stall_per_inst")
    occ_avg = runtime_profile.get("occupancy_average_active")
    avg_stall = runtime_profile.get("avg_stall_per_inst")
    vgpr_count = (focus_resource or {}).get("vgpr_count")

    if isinstance(sync_wait_share, (int, float)) and sync_wait_share >= 0.25:
        add_hint("sync_wait_share", sync_wait_share, 0.25)
    if isinstance(immed_stall_per_inst, (int, float)) and immed_stall_per_inst >= 10.0:
        add_hint("immed_stall_per_inst", immed_stall_per_inst, 10.0)
    if memory_access_hints.get("global_memory_bound_suspected") and isinstance(global_mem_duration_share, (int, float)):
        add_hint("global_memory_duration_share", global_mem_duration_share, 0.05)
    if memory_access_hints.get("lds_pressure_suspected") and isinstance(lds_stall_per_inst, (int, float)):
        add_hint("lds_stall_per_inst", lds_stall_per_inst, 0.10)
    if isinstance(avg_stall, (int, float)) and isinstance(occ_avg, (int, float)) and isinstance(vgpr_count, (int, float)):
        if avg_stall >= 1.0 and vgpr_count >= 32:
            add_hint("vgpr_count", vgpr_count, 32)

    hints.sort(
        key=lambda item: (
            {"sync_wait_share": 0, "immed_stall_per_inst": 1, "global_memory_duration_share": 2, "lds_stall_per_inst": 3, "vgpr_count": 4}.get(item["name"], 99),
            -float(item.get("value") or 0.0),
        )
    )
    return hints[:5]


def build_occupancy_detail(
    runtime_profile: dict[str, Any],
    focus_resource: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "runtime_average_active": runtime_profile.get("occupancy_average_active"),
        "runtime_max_active": runtime_profile.get("occupancy_max_active"),
        "avg_wave_lifetime": runtime_profile.get("avg_wave_lifetime"),
        "max_wave_lifetime": runtime_profile.get("max_wave_lifetime"),
        "stalled_instruction_share": runtime_profile.get("stalled_instruction_share"),
        "vgpr_count": (focus_resource or {}).get("vgpr_count"),
        "lds_size": (focus_resource or {}).get("lds_size"),
        "wavefront_size": (focus_resource or {}).get("wavefront_size"),
    }


def build_event_barrier_context(stitch: dict[str, Any]) -> dict[str, Any]:
    dispatch_spans = int(stitch.get("dispatch_api_span_count", 0) or 0)
    command_buffer_spans = int(stitch.get("command_buffer_span_count", 0) or 0)
    barrier_spans = int(stitch.get("barrier_span_count", 0) or 0)
    return {
        "dispatch_api_span_count": dispatch_spans,
        "dispatch_span_assignment_count": int(stitch.get("dispatch_span_assignment_count", 0) or 0),
        "bind_marker_count": int(stitch.get("bind_marker_count", 0) or 0),
        "command_buffer_span_count": command_buffer_spans,
        "barrier_marker_count": int(stitch.get("barrier_marker_count", 0) or 0),
        "barrier_span_count": barrier_spans,
        "unmatched_barrier_begin_count": int(stitch.get("unmatched_barrier_begin_count", 0) or 0),
        "dispatches_per_cb": (dispatch_spans / command_buffer_spans) if command_buffer_spans else None,
        "barriers_per_dispatch": (barrier_spans / dispatch_spans) if dispatch_spans else None,
    }


def build_tuning_summary(
    *,
    bottleneck_hints: list[dict[str, Any]],
    event_barrier_context: dict[str, Any],
    instruction_ranking: list[dict[str, Any]],
    source_hints: dict[str, Any],
) -> dict[str, Any]:
    metric_rows = [
        {
            "name": item.get("name"),
            "value": item.get("value"),
            "threshold": item.get("threshold"),
        }
        for item in bottleneck_hints[:3]
        if item.get("name")
    ]
    hot_pcs = [f"0x{int(item.get('pc', 0) or 0):x}" for item in instruction_ranking[:4]]
    source_lines = [int(item.get("line", 0) or 0) for item in (source_hints.get("matches") or [])[:6] if item.get("line") is not None]
    return {
        "metric_rows": metric_rows,
        "dispatches_per_cb": event_barrier_context.get("dispatches_per_cb"),
        "barriers_per_dispatch": event_barrier_context.get("barriers_per_dispatch"),
        "hot_pcs": hot_pcs,
        "source_lines": source_lines,
    }


def build_tuning_summary_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_metric_rows": baseline.get("metric_rows") or [],
        "candidate_metric_rows": candidate.get("metric_rows") or [],
        "baseline_hot_pcs": baseline.get("hot_pcs") or [],
        "candidate_hot_pcs": candidate.get("hot_pcs") or [],
        "baseline_source_lines": baseline.get("source_lines") or [],
        "candidate_source_lines": candidate.get("source_lines") or [],
    }


def build_bottleneck_hint_deltas(
    baseline_hints: list[dict[str, Any]],
    candidate_hints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before = {str(item.get("name")): item for item in baseline_hints}
    after = {str(item.get("name")): item for item in candidate_hints}
    rows: list[dict[str, Any]] = []
    for name in sorted(set(before) | set(after)):
        before_item = before.get(name) or {}
        after_item = after.get(name) or {}
        rows.append(
            {
                "name": name,
                "before": before_item.get("value"),
                "after": after_item.get("value"),
                "delta": numeric_delta(before_item.get("value"), after_item.get("value")),
                "baseline_present": name in before,
                "candidate_present": name in after,
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item.get("baseline_present") != item.get("candidate_present") else 1,
            -abs(((item.get("delta") or {}).get("delta") or 0)),
            item["name"],
        )
    )
    return rows[:5]


def _pick(data: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(data, dict):
        return None
    return data.get(key)
