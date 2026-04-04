from __future__ import annotations

from typing import Any


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def _render_region_proxy_metric(item: dict[str, Any] | None) -> str:
    if not item:
        return "None"
    return (
        f"metric={item.get('metric')} "
        f"value={_fmt_float(item.get('value'))} "
        f"weight={_fmt_float(item.get('weight'))} "
        f"weighted={_fmt_float(item.get('weighted_value'))}"
    )


def render_region_focus_payload(payload: dict[str, Any], *, source_excerpt: bool = False) -> str:
    lines = ["region_focus:"]
    lines.append(f"  file: {payload.get('file')}")
    lines.append(f"  focus_code_object: {payload.get('focus_code_object_index')}")
    trace_quality = payload.get("trace_quality") or {}
    profiling_constraints = payload.get("profiling_constraints") or {}
    lines.append(
        "  trace_quality: "
        f"level={trace_quality.get('runtime_evidence_level')} "
        f"submit_dilution_suspected={profiling_constraints.get('submit_dilution_suspected')}"
    )
    resource = payload.get("resource") or {}
    if resource:
        lines.append(
            "  resource: "
            f"entry_point={resource.get('entry_point')} vgpr={resource.get('vgpr_count')} "
            f"sgpr={resource.get('sgpr_count')} lds={resource.get('lds_size')} "
            f"scratch={resource.get('scratch_memory_size')} wavefront={resource.get('wavefront_size')}"
        )
    source_regions = payload.get("source_region_metrics") or []
    if source_regions:
        lines.append("  source_regions:")
        for item in source_regions[:8]:
            lines.append(
                "    - "
                f"label={item.get('label')} "
                f"attribution_mode={item.get('attribution_mode')} "
                f"proxy_score={_fmt_float(item.get('proxy_score'))} "
                f"hotspot_duration={item.get('hotspot_duration')} "
                f"hotspot_stall={item.get('hotspot_stall')}"
            )
            for metric_item in (item.get("proxy_metric_rows") or [])[:4]:
                lines.append(
                    "      "
                    f"proxy_metric={metric_item.get('metric')} "
                    f"value={_fmt_float(metric_item.get('value'))} "
                    f"weight={_fmt_float(metric_item.get('weight'))} "
                    f"weighted={_fmt_float(metric_item.get('weighted_value'))}"
                )
            for line_no in (item.get("source_lines") or [])[:6]:
                lines.append(f"      source_line={line_no}")
            for pc in (item.get("top_pcs") or [])[:6]:
                lines.append(f"      hot_pc={pc}")
    if source_excerpt:
        source_hints = payload.get("source_hints") or {}
        if source_hints:
            lines.append("  source_hints:")
            for item in (source_hints.get("matches") or [])[:8]:
                lines.append(f"    - line={item.get('line')} match={item.get('match')}")
                for excerpt in (item.get("excerpt") or [])[:3]:
                    prefix = ">" if excerpt.get("focus") else " "
                    lines.append(f"      {prefix}L{excerpt.get('line')}: {excerpt.get('text')}")
    return "\n".join(lines)


def render_compare_region_focus_payload(payload: dict[str, Any], *, source_excerpt: bool = False) -> str:
    lines = ["compare_region_focus:"]
    lines.append(f"  baseline: {payload.get('baseline_file')}")
    lines.append(f"  candidate: {payload.get('candidate_file')}")
    trace = payload.get("trace_quality") or {}
    lines.append(
        "  trace_quality: "
        f"baseline={trace.get('baseline')} candidate={trace.get('candidate')} "
        f"baseline_submit_dilution={trace.get('baseline_submit_dilution_suspected')} "
        f"candidate_submit_dilution={trace.get('candidate_submit_dilution_suspected')}"
    )
    source_region_deltas = payload.get("source_region_deltas") or []
    if source_region_deltas:
        lines.append("  source_region_deltas:")
        for item in source_region_deltas[:8]:
            lines.append(
                "    - "
                f"label={item.get('label')} "
                f"attribution_mode={item.get('attribution_mode_before')} -> {item.get('attribution_mode_after')} "
                f"proxy_score={_fmt_float(item.get('proxy_score_before'))} -> {_fmt_float(item.get('proxy_score_after'))} "
                f"hotspot_duration={item.get('hotspot_duration_before')} -> {item.get('hotspot_duration_after')} "
                f"hotspot_stall={item.get('hotspot_stall_before')} -> {item.get('hotspot_stall_after')}"
            )
            before_metric_rows = item.get("proxy_metric_rows_before") or []
            after_metric_rows = item.get("proxy_metric_rows_after") or []
            for idx in range(max(len(before_metric_rows), len(after_metric_rows))):
                before = before_metric_rows[idx] if idx < len(before_metric_rows) else None
                after = after_metric_rows[idx] if idx < len(after_metric_rows) else None
                lines.append(
                    "      "
                    f"proxy_metric[{idx}] "
                    f"baseline={_render_region_proxy_metric(before)} "
                    f"candidate={_render_region_proxy_metric(after)}"
                )
            before_lines = item.get("source_lines_before") or []
            after_lines = item.get("source_lines_after") or []
            for idx in range(max(len(before_lines), len(after_lines))):
                before = before_lines[idx] if idx < len(before_lines) else None
                after = after_lines[idx] if idx < len(after_lines) else None
                lines.append(f"      source_line[{idx}] baseline={before} candidate={after}")
            before_pcs = item.get("top_pcs_before") or []
            after_pcs = item.get("top_pcs_after") or []
            for idx in range(max(len(before_pcs), len(after_pcs))):
                before = before_pcs[idx] if idx < len(before_pcs) else None
                after = after_pcs[idx] if idx < len(after_pcs) else None
                lines.append(f"      hot_pc[{idx}] baseline={before} candidate={after}")
    if source_excerpt:
        source_delta_hints = payload.get("source_delta_hints") or []
        if source_delta_hints:
            lines.append("  source_delta_hints:")
            for item in source_delta_hints[:8]:
                lines.append(f"    - line={item.get('line')} match={item.get('match')}")
                for excerpt in (item.get("excerpt") or [])[:3]:
                    prefix = ">" if excerpt.get("focus") else " "
                    lines.append(f"      {prefix}L{excerpt.get('line')}: {excerpt.get('text')}")
    return "\n".join(lines)
