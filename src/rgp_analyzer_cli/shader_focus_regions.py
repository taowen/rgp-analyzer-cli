from __future__ import annotations

from typing import Any


def build_source_region_metrics(
    source_isa_blocks: list[dict[str, Any]],
    instruction_ranking: list[dict[str, Any]],
    hotspot_candidates: list[dict[str, Any]],
    runtime_proxies: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not source_isa_blocks:
        return []

    runtime_proxies = runtime_proxies or {}
    instruction_by_pc = {
        int(item.get("pc", 0) or 0): item
        for item in instruction_ranking
        if item.get("pc") is not None
    }

    rows: list[dict[str, Any]] = []
    has_runtime_attribution = bool(hotspot_candidates or instruction_ranking)
    for block in source_isa_blocks:
        pcs = [int(item.get("pc", 0) or 0) for item in (block.get("isa_instructions") or [])]
        pc_set = set(pcs)
        dispatch_count = 0
        score = 0
        hotspot_mentions = 0
        hotspot_duration = 0
        hotspot_stall = 0

        for pc in pcs:
            item = instruction_by_pc.get(pc) or {}
            dispatch_count += int(item.get("dispatch_count", 0) or 0)
            score += int(item.get("score", 0) or 0)

        for hotspot in hotspot_candidates:
            hotspot_pcs = {
                int(item.get("pc", 0) or 0)
                for item in (hotspot.get("top_pcs") or [])
                if item.get("pc") is not None
            }
            if hotspot_pcs & pc_set:
                hotspot_mentions += 1
                hotspot_duration += int(hotspot.get("total_duration", 0) or 0)
                hotspot_stall += int(hotspot.get("total_stall", 0) or 0)

        proxy_metric_rows = _region_proxy_metric_rows(str(block.get("label") or ""), runtime_proxies)
        proxy_score = sum(float(item.get("weighted_value", 0.0) or 0.0) for item in proxy_metric_rows)
        source_lines = [int(item.get("line", 0) or 0) for item in (block.get("source_lines") or [])]
        rows.append(
            {
                "label": block.get("label"),
                "attribution_mode": "runtime" if has_runtime_attribution else "proxy",
                "pc_count": len(pc_set),
                "dispatch_count": dispatch_count,
                "score": score,
                "hotspot_mentions": hotspot_mentions,
                "hotspot_duration": hotspot_duration,
                "hotspot_stall": hotspot_stall,
                "avg_hotspot_stall": (hotspot_stall / hotspot_mentions) if hotspot_mentions else 0.0,
                "proxy_score": proxy_score,
                "proxy_metric_rows": proxy_metric_rows,
                "source_lines": source_lines,
                "top_pcs": [f"0x{pc:x}" for pc in pcs[:4]],
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("hotspot_duration", 0) or 0),
            -float(item.get("proxy_score", 0.0) or 0.0),
            -int(item.get("score", 0) or 0),
            -int(item.get("dispatch_count", 0) or 0),
            str(item.get("label") or ""),
        )
    )
    return rows[:8]


def build_source_region_deltas(
    baseline_regions: list[dict[str, Any]],
    candidate_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_map = {str(item.get("label")): item for item in baseline_regions}
    candidate_map = {str(item.get("label")): item for item in candidate_regions}
    labels = sorted(set(baseline_map) | set(candidate_map))

    rows: list[dict[str, Any]] = []
    for label in labels:
        baseline = baseline_map.get(label) or {}
        candidate = candidate_map.get(label) or {}
        rows.append(
            {
                "label": label,
                "attribution_mode_before": baseline.get("attribution_mode"),
                "attribution_mode_after": candidate.get("attribution_mode"),
                "dispatch_count_before": baseline.get("dispatch_count"),
                "dispatch_count_after": candidate.get("dispatch_count"),
                "score_before": baseline.get("score"),
                "score_after": candidate.get("score"),
                "hotspot_duration_before": baseline.get("hotspot_duration"),
                "hotspot_duration_after": candidate.get("hotspot_duration"),
                "hotspot_stall_before": baseline.get("hotspot_stall"),
                "hotspot_stall_after": candidate.get("hotspot_stall"),
                "avg_hotspot_stall_before": baseline.get("avg_hotspot_stall"),
                "avg_hotspot_stall_after": candidate.get("avg_hotspot_stall"),
                "proxy_score_before": baseline.get("proxy_score"),
                "proxy_score_after": candidate.get("proxy_score"),
                "proxy_metric_rows_before": baseline.get("proxy_metric_rows") or [],
                "proxy_metric_rows_after": candidate.get("proxy_metric_rows") or [],
                "source_lines_before": baseline.get("source_lines") or [],
                "source_lines_after": candidate.get("source_lines") or [],
                "top_pcs_before": baseline.get("top_pcs") or [],
                "top_pcs_after": candidate.get("top_pcs") or [],
            }
        )

    rows.sort(
        key=lambda item: (
            -abs(int((item.get("hotspot_duration_after") or 0)) - int((item.get("hotspot_duration_before") or 0))),
            -abs(int((item.get("hotspot_stall_after") or 0)) - int((item.get("hotspot_stall_before") or 0))),
            -abs(float((item.get("proxy_score_after") or 0.0)) - float((item.get("proxy_score_before") or 0.0))),
            str(item.get("label") or ""),
        )
    )
    return rows[:8]


def _region_proxy_metric_rows(label: str, runtime_proxies: dict[str, Any]) -> list[dict[str, Any]]:
    weight_rows = _region_proxy_weights(label)
    rows: list[dict[str, Any]] = []
    for metric_name, weight in weight_rows:
        value = runtime_proxies.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        rows.append(
            {
                "metric": metric_name,
                "value": float(value),
                "weight": float(weight),
                "weighted_value": float(value) * float(weight),
            }
        )
    rows.sort(key=lambda item: (-abs(float(item.get("weighted_value", 0.0) or 0.0)), str(item.get("metric") or "")))
    return rows[:4]


def _region_proxy_weights(label: str) -> list[tuple[str, float]]:
    mapping: dict[str, list[tuple[str, float]]] = {
        "tile_staging": [
            ("global_memory_duration_share", 0.50),
            ("lds_duration_share", 0.30),
            ("sync_wait_cycles_per_inst", 0.20),
        ],
        "quantized_qk": [
            ("vector_duration_share", 0.35),
            ("scalar_duration_share", 0.20),
            ("global_memory_duration_share", 0.20),
            ("immed_stall_per_inst", 0.25),
        ],
        "softmax_update": [
            ("sync_wait_share", 0.35),
            ("sync_wait_cycles_per_inst", 0.30),
            ("immed_stall_per_inst", 0.25),
            ("scalar_duration_share", 0.10),
        ],
        "value_accumulate": [
            ("vector_duration_share", 0.45),
            ("global_memory_duration_share", 0.20),
            ("sync_wait_cycles_per_inst", 0.15),
            ("immed_stall_per_inst", 0.20),
        ],
        "shared_exchange": [
            ("lds_duration_share", 0.30),
            ("lds_stall_per_inst", 0.30),
            ("sync_wait_share", 0.20),
            ("immed_stall_per_inst", 0.20),
        ],
        "global_load": [
            ("global_memory_instruction_share", 0.40),
            ("global_memory_duration_share", 0.40),
            ("global_memory_stall_share", 0.20),
        ],
        "cooperative_matrix": [
            ("vector_duration_share", 0.50),
            ("sync_wait_cycles_per_inst", 0.20),
            ("immed_stall_per_inst", 0.10),
            ("global_memory_duration_share", 0.20),
        ],
        "bounds_check": [
            ("scalar_duration_share", 0.40),
            ("immed_stall_per_inst", 0.30),
            ("sync_wait_share", 0.30),
        ],
        "buffer_store": [
            ("global_memory_duration_share", 0.50),
            ("immed_stall_per_inst", 0.20),
            ("sync_wait_cycles_per_inst", 0.30),
        ],
        "kernel_end": [
            ("sync_wait_share", 0.50),
            ("immed_stall_per_inst", 0.50),
        ],
    }
    return mapping.get(
        label,
        [
            ("vector_duration_share", 0.25),
            ("scalar_duration_share", 0.25),
            ("sync_wait_cycles_per_inst", 0.25),
            ("immed_stall_per_inst", 0.25),
        ],
    )
