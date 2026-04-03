from __future__ import annotations

from typing import Any


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("summary") or {}


def _pick(data: dict[str, Any], key: str) -> Any:
    return data.get(key) if isinstance(data, dict) else None


def _numeric_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    delta = after - before
    payload: dict[str, Any] = {"before": before, "after": after, "delta": delta}
    if before not in (0, 0.0):
        payload["delta_ratio"] = delta / before
    return payload


def _field_delta(before: dict[str, Any], after: dict[str, Any], *fields: str) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for field in fields:
        delta = _numeric_delta(_pick(before, field), _pick(after, field))
        if delta is not None:
            deltas[field] = delta
    return deltas


def _hotspot_pc_signature(item: dict[str, Any]) -> str | None:
    address = item.get("address")
    symbol = item.get("symbol") or {}
    if symbol.get("name"):
        return f"{symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
    if address is not None:
        return f"0x{int(address or 0):x}"
    return None


def _dispatch_pc_signature(item: dict[str, Any]) -> str | None:
    if item.get("pc") is None:
        return None
    return (
        f"code_object[{int(item.get('code_object_index', 0) or 0)}]"
        f":0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic') or ''} {item.get('operands') or ''}".strip()
    )


def _ranking_compare(
    baseline_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    *,
    signature_fn,
    metric_key: str,
) -> list[dict[str, Any]]:
    baseline_by_sig = {
        signature_fn(item): item for item in baseline_items if signature_fn(item)
    }
    candidate_by_sig = {
        signature_fn(item): item for item in candidate_items if signature_fn(item)
    }
    signatures = set(baseline_by_sig) | set(candidate_by_sig)
    rows: list[dict[str, Any]] = []
    for signature in signatures:
        before_item = baseline_by_sig.get(signature) or {}
        after_item = candidate_by_sig.get(signature) or {}
        before = before_item.get(metric_key, 0) or 0
        after = after_item.get(metric_key, 0) or 0
        rows.append(
            {
                "signature": signature,
                "before": before,
                "after": after,
                "delta": after - before,
            }
        )
    rows.sort(key=lambda item: (-abs(item["delta"]), -item["after"], str(item["signature"])))
    return rows[:8]


def _hotspot_signature(hotspot: dict[str, Any] | None) -> str | None:
    if not hotspot:
        return None
    symbol = hotspot.get("symbol") or {}
    name = symbol.get("name")
    offset = symbol.get("offset")
    if name:
        return f"{name}+0x{int(offset or 0):x}"
    address = hotspot.get("address")
    if address is not None:
        return f"0x{int(address or 0):x}"
    return None


def compare_triage_payloads(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = _summary(baseline)
    candidate_summary = _summary(candidate)

    baseline_trace = baseline_summary.get("trace_quality") or {}
    candidate_trace = candidate_summary.get("trace_quality") or {}
    baseline_resource = baseline_summary.get("resource") or {}
    candidate_resource = candidate_summary.get("resource") or {}
    baseline_runtime = baseline_summary.get("runtime") or {}
    candidate_runtime = candidate_summary.get("runtime") or {}
    baseline_decoder = baseline_summary.get("decoder") or {}
    candidate_decoder = candidate_summary.get("decoder") or {}
    baseline_dispatch = baseline_summary.get("dispatch_isa") or {}
    candidate_dispatch = candidate_summary.get("dispatch_isa") or {}
    baseline_hotspot = baseline_summary.get("top_hotspot") or {}
    candidate_hotspot = candidate_summary.get("top_hotspot") or {}
    baseline_constraints = baseline_summary.get("profiling_constraints") or {}
    candidate_constraints = candidate_summary.get("profiling_constraints") or {}
    baseline_category_counts = baseline_runtime.get("category_counts") or {}
    candidate_category_counts = candidate_runtime.get("category_counts") or {}
    category_count_deltas = _field_delta(
        baseline_category_counts,
        candidate_category_counts,
        "VALU",
        "SALU",
        "LDS",
        "VMEM",
        "SMEM",
    )

    return {
        "baseline_file": baseline.get("file"),
        "candidate_file": candidate.get("file"),
        "trace_quality": {
            "baseline": baseline_trace.get("runtime_evidence_level"),
            "candidate": candidate_trace.get("runtime_evidence_level"),
            "baseline_submit_dilution_suspected": baseline_constraints.get("submit_dilution_suspected"),
            "candidate_submit_dilution_suspected": candidate_constraints.get("submit_dilution_suspected"),
            "deltas": _field_delta(
                baseline_trace,
                candidate_trace,
                "sqtt_trace_bytes",
                "queue_event_count",
                "decoded_instruction_count",
                "decoded_wave_count",
                "dispatch_span_count",
                "mapped_dispatch_count",
                "total_dispatch_count",
            ),
        },
        "resource": {
            "entry_point": candidate_resource.get("entry_point") or baseline_resource.get("entry_point"),
            "deltas": _field_delta(
                baseline_resource,
                candidate_resource,
                "vgpr_count",
                "sgpr_count",
                "lds_size",
                "scratch_memory_size",
                "wavefront_size",
            ),
        },
        "runtime": {
            "deltas": _field_delta(
                baseline_runtime,
                candidate_runtime,
                "instructions",
                "waves",
                "avg_stall_per_inst",
                "stall_share_of_duration",
                "occupancy_average_active",
                "occupancy_max_active",
            ),
            "baseline_top_category": (baseline_runtime.get("top_category") or {}).get("category"),
            "candidate_top_category": (candidate_runtime.get("top_category") or {}).get("category"),
            "baseline_top_wave_state": (baseline_runtime.get("top_wave_state") or {}).get("state"),
            "candidate_top_wave_state": (candidate_runtime.get("top_wave_state") or {}).get("state"),
            "category_count_deltas": category_count_deltas,
            "hotspot_ranking_delta": _ranking_compare(
                baseline_runtime.get("top_hotspot_profiles") or [],
                candidate_runtime.get("top_hotspot_profiles") or [],
                signature_fn=_hotspot_pc_signature,
                metric_key="total_duration",
            ),
        },
        "decoder": {
            "deltas": _field_delta(
                baseline_decoder,
                candidate_decoder,
                "code_object_count",
                "code_object_load_failures",
                "dispatch_isa_mapped",
                "dispatch_isa_total",
            ),
            "baseline_sparse_runtime_trace": baseline_decoder.get("sparse_runtime_trace"),
            "candidate_sparse_runtime_trace": candidate_decoder.get("sparse_runtime_trace"),
        },
        "dispatch_isa": {
            "deltas": _field_delta(
                baseline_dispatch,
                candidate_dispatch,
                "mapped_dispatch_count",
                "dispatch_count",
            ),
            "baseline_top_pc": baseline_dispatch.get("top_pc"),
            "candidate_top_pc": candidate_dispatch.get("top_pc"),
            "pc_ranking_delta": _ranking_compare(
                baseline_dispatch.get("top_pcs") or [],
                candidate_dispatch.get("top_pcs") or [],
                signature_fn=_dispatch_pc_signature,
                metric_key="count",
            ),
        },
        "hotspot": {
            "baseline": {
                "signature": _hotspot_signature(baseline_hotspot),
                "code_object_index": baseline_hotspot.get("code_object_index"),
                "match_kind": baseline_hotspot.get("match_kind"),
            },
            "candidate": {
                "signature": _hotspot_signature(candidate_hotspot),
                "code_object_index": candidate_hotspot.get("code_object_index"),
                "match_kind": candidate_hotspot.get("match_kind"),
            },
            "deltas": _field_delta(
                baseline_hotspot,
                candidate_hotspot,
                "total_duration",
                "total_stall",
                "hitcount",
                "candidate_count",
                "dispatch_assignment_share",
                "dispatch_isa_share",
            ),
            "profile_deltas": _field_delta(
                baseline_runtime.get("top_hotspot_profile") or {},
                candidate_runtime.get("top_hotspot_profile") or {},
                "avg_duration_per_hit",
                "avg_stall_per_hit",
                "stall_share_of_duration",
            ),
        },
    }
