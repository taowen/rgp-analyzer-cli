from __future__ import annotations

from .render_focus import _fmt_float, _fmt_metric_row, _fmt_ratio_pct


def render_compare_shader_focus_payload(payload: dict[str, object], *, source_excerpt: bool = False) -> str:
    lines = ["compare_shader_focus:"]
    lines.append(f"  baseline: {payload.get('baseline_file')}")
    lines.append(f"  candidate: {payload.get('candidate_file')}")
    focus = payload.get("focus_code_object_index") or {}
    lines.append(
        "  focus_code_object: "
        f"baseline={focus.get('baseline')} candidate={focus.get('candidate')}"
    )
    source_files = payload.get("source_files") or {}
    if source_files:
        lines.append(
            "  source_files: "
            f"baseline={source_files.get('baseline')} "
            f"candidate={source_files.get('candidate')}"
        )
    source_delta_hints = payload.get("source_delta_hints") or []
    if source_delta_hints:
        lines.append("  source_delta_hints:")
        for item in source_delta_hints[:6]:
            lines.append(f"    - line={item.get('line')} match={item.get('match')}")
            if source_excerpt:
                for excerpt in (item.get("excerpt") or [])[:3]:
                    prefix = ">" if excerpt.get("focus") else " "
                    lines.append(f"      {prefix}L{excerpt.get('line')}: {excerpt.get('text')}")
    bottleneck_hint_deltas = payload.get("bottleneck_hint_deltas") or []
    if bottleneck_hint_deltas:
        lines.append("  bottleneck_metric_deltas:")
        for item in bottleneck_hint_deltas[:5]:
            delta = item.get("delta") or {}
            delta_text = ""
            if delta:
                delta_text = (
                    f" value={_fmt_float(delta.get('before'))} -> {_fmt_float(delta.get('after'))} "
                    f"(delta={_fmt_float(delta.get('delta'))})"
                )
            lines.append(
                "    - "
                f"metric={item.get('name')} baseline_present={item.get('baseline_present')} "
                f"candidate_present={item.get('candidate_present')}{delta_text}"
            )
    occupancy_detail_deltas = payload.get("occupancy_detail_deltas") or {}
    if occupancy_detail_deltas:
        lines.append("  occupancy_detail_deltas:")
        for key, item in occupancy_detail_deltas.items():
            line = f"    - {key}: {item.get('before')} -> {item.get('after')} (delta={item.get('delta')}"
            ratio = item.get("delta_ratio")
            if isinstance(ratio, (int, float)):
                line += f", ratio={ratio:.3f}"
            line += ")"
            lines.append(line)
    event_barrier_context_deltas = payload.get("event_barrier_context_deltas") or {}
    if event_barrier_context_deltas:
        lines.append("  event_barrier_context_deltas:")
        for key, item in event_barrier_context_deltas.items():
            line = f"    - {key}: {item.get('before')} -> {item.get('after')} (delta={item.get('delta')}"
            ratio = item.get("delta_ratio")
            if isinstance(ratio, (int, float)):
                line += f", ratio={ratio:.3f}"
            line += ")"
            lines.append(line)
    tuning_summary_delta = payload.get("tuning_summary_delta") or {}
    if tuning_summary_delta:
        lines.append("  tuning_summary_delta:")
        baseline_metric_rows = tuning_summary_delta.get("baseline_metric_rows") or []
        candidate_metric_rows = tuning_summary_delta.get("candidate_metric_rows") or []
        max_rows = max(len(baseline_metric_rows), len(candidate_metric_rows))
        for idx in range(min(max_rows, 4)):
            baseline_item = baseline_metric_rows[idx] if idx < len(baseline_metric_rows) else {}
            candidate_item = candidate_metric_rows[idx] if idx < len(candidate_metric_rows) else {}
            lines.append(
                "    - "
                f"baseline[{idx}] {_fmt_metric_row(baseline_item)} "
                f"candidate[{idx}] {_fmt_metric_row(candidate_item)}"
            )
        baseline_hot_pcs = tuning_summary_delta.get("baseline_hot_pcs") or []
        candidate_hot_pcs = tuning_summary_delta.get("candidate_hot_pcs") or []
        for idx in range(min(max(len(baseline_hot_pcs), len(candidate_hot_pcs)), 4)):
            before = baseline_hot_pcs[idx] if idx < len(baseline_hot_pcs) else None
            after = candidate_hot_pcs[idx] if idx < len(candidate_hot_pcs) else None
            lines.append(f"    - hot_pc[{idx}] baseline={before} candidate={after}")
        baseline_source_lines = tuning_summary_delta.get("baseline_source_lines") or []
        candidate_source_lines = tuning_summary_delta.get("candidate_source_lines") or []
        for idx in range(min(max(len(baseline_source_lines), len(candidate_source_lines)), 6)):
            before = baseline_source_lines[idx] if idx < len(baseline_source_lines) else None
            after = candidate_source_lines[idx] if idx < len(candidate_source_lines) else None
            lines.append(f"    - source_line[{idx}] baseline={before} candidate={after}")
    enough = payload.get("enough_for_shader_tuning") or {}
    lines.append(
        "  enough_for_shader_tuning: "
        f"baseline={enough.get('baseline')} candidate={enough.get('candidate')}"
    )
    trace = payload.get("trace_quality") or {}
    lines.append(
        "  trace_quality: "
        f"baseline={trace.get('baseline')} candidate={trace.get('candidate')} "
        f"baseline_submit_dilution={trace.get('baseline_submit_dilution_suspected')} "
        f"candidate_submit_dilution={trace.get('candidate_submit_dilution_suspected')}"
    )
    for section_name in (
        "resource_deltas",
        "runtime_deltas",
        "runtime_category_count_deltas",
        "runtime_category_duration_deltas",
        "runtime_category_stall_deltas",
        "runtime_proxy_deltas",
        "hotspot_deltas",
        "hotspot_profile_deltas",
        "dispatch_isa_deltas",
    ):
        section = payload.get(section_name) or {}
        if section:
            lines.append(f"  {section_name}:")
            for key, item in section.items():
                line = f"    - {key}: {item.get('before')} -> {item.get('after')} (delta={item.get('delta')}"
                ratio = item.get("delta_ratio")
                if isinstance(ratio, (int, float)):
                    line += f", ratio={ratio:.3f}"
                line += ")"
                lines.append(line)
    top_pcs = payload.get("dispatch_isa_pc_ranking_delta") or []
    if top_pcs:
        lines.append("  dispatch_isa_pc_ranking_delta:")
        for item in top_pcs[:4]:
            lines.append(
                f"    - {item.get('signature')}: {item.get('before')} -> {item.get('after')} "
                f"(delta={item.get('delta')})"
            )
    runtime_hotspot_ranking = payload.get("runtime_hotspot_ranking_delta") or []
    if runtime_hotspot_ranking:
        lines.append("  runtime_hotspot_ranking_delta:")
        for item in runtime_hotspot_ranking[:4]:
            lines.append(
                f"    - {item.get('signature')}: {item.get('before')} -> {item.get('after')} "
                f"(delta={item.get('delta')})"
            )
    instruction_ranking = payload.get("instruction_ranking_delta") or []
    if instruction_ranking:
        lines.append("  instruction_ranking_delta:")
        for item in instruction_ranking[:6]:
            score = item.get("score") or {}
            dispatch_count = item.get("dispatch_count") or {}
            hotspot_mentions = item.get("hotspot_mentions") or {}
            hotspot_duration = item.get("hotspot_duration") or {}
            hotspot_stall = item.get("hotspot_stall") or {}
            parts = []
            if score:
                parts.append(
                    f"score {score.get('before')} -> {score.get('after')} "
                    f"(delta={score.get('delta')})"
                )
            if dispatch_count:
                parts.append(
                    f"dispatch_count {dispatch_count.get('before')} -> {dispatch_count.get('after')} "
                    f"(delta={dispatch_count.get('delta')})"
                )
            if hotspot_mentions:
                parts.append(
                    f"hotspot_mentions {hotspot_mentions.get('before')} -> {hotspot_mentions.get('after')} "
                    f"(delta={hotspot_mentions.get('delta')})"
                )
            if hotspot_duration:
                parts.append(
                    f"hotspot_duration {hotspot_duration.get('before')} -> {hotspot_duration.get('after')} "
                    f"(delta={hotspot_duration.get('delta')})"
                )
            if hotspot_stall:
                parts.append(
                    f"hotspot_stall {hotspot_stall.get('before')} -> {hotspot_stall.get('after')} "
                    f"(delta={hotspot_stall.get('delta')})"
                )
            line = f"    - {item.get('signature')}: " + "; ".join(parts)
            if item.get("text"):
                line += f" isa=\"{item.get('text')}\""
            lines.append(line)
    source_region_deltas = payload.get("source_region_deltas") or []
    if source_region_deltas:
        lines.append("  source_region_deltas:")
        for item in source_region_deltas[:6]:
            lines.append(
                "    - "
                f"label={item.get('label')} "
                f"attribution_mode={item.get('attribution_mode_before')} -> {item.get('attribution_mode_after')} "
                f"hotspot_duration={item.get('hotspot_duration_before')} -> {item.get('hotspot_duration_after')} "
                f"hotspot_stall={item.get('hotspot_stall_before')} -> {item.get('hotspot_stall_after')} "
                f"avg_hotspot_stall={_fmt_float(item.get('avg_hotspot_stall_before'))} -> {_fmt_float(item.get('avg_hotspot_stall_after'))} "
                f"score={item.get('score_before')} -> {item.get('score_after')} "
                f"proxy_score={_fmt_float(item.get('proxy_score_before'))} -> {_fmt_float(item.get('proxy_score_after'))}"
            )
            before_metric_rows = item.get("proxy_metric_rows_before") or []
            after_metric_rows = item.get("proxy_metric_rows_after") or []
            for idx in range(max(len(before_metric_rows), len(after_metric_rows))):
                before = before_metric_rows[idx] if idx < len(before_metric_rows) else None
                after = after_metric_rows[idx] if idx < len(after_metric_rows) else None
                line = f"      proxy_metric[{idx}] "
                if before:
                    line += (
                        f"baseline=metric={before.get('metric')} value={_fmt_float(before.get('value'))} "
                        f"weight={_fmt_float(before.get('weight'))} weighted={_fmt_float(before.get('weighted_value'))}"
                    )
                else:
                    line += "baseline=None"
                if after:
                    line += (
                        f" candidate=metric={after.get('metric')} value={_fmt_float(after.get('value'))} "
                        f"weight={_fmt_float(after.get('weight'))} weighted={_fmt_float(after.get('weighted_value'))}"
                    )
                else:
                    line += " candidate=None"
                lines.append(line)
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
    focused_runtime_hotspots = payload.get("focused_runtime_hotspot_deltas") or []
    if focused_runtime_hotspots:
        lines.append("  focused_runtime_hotspot_deltas:")
        for item in focused_runtime_hotspots[:3]:
            duration = item.get("duration") or {}
            stall = item.get("stall") or {}
            avg_duration = item.get("avg_duration_per_hit") or {}
            avg_stall = item.get("avg_stall_per_hit") or {}
            line = f"    - {item.get('signature')}:"
            parts = []
            if duration:
                parts.append(f"duration {duration.get('before')} -> {duration.get('after')} (delta={duration.get('delta')})")
            if stall:
                parts.append(f"stall {stall.get('before')} -> {stall.get('after')} (delta={stall.get('delta')})")
            if avg_duration:
                parts.append(f"avg_duration {_fmt_float(avg_duration.get('before'))} -> {_fmt_float(avg_duration.get('after'))}")
            if avg_stall:
                parts.append(f"avg_stall {_fmt_float(avg_stall.get('before'))} -> {_fmt_float(avg_stall.get('after'))}")
            lines.append(line + " " + "; ".join(parts))
            top_pcs = item.get("top_pcs") or []
            for pc in top_pcs[:3]:
                lines.append(
                    f"      pc_delta {pc.get('signature')}: {pc.get('before')} -> {pc.get('after')} "
                    f"(delta={pc.get('delta')})"
                )
    runtime_deltas = payload.get("runtime_deltas") or {}
    category_deltas = payload.get("runtime_category_count_deltas") or {}
    runtime_proxy_deltas = payload.get("runtime_proxy_deltas") or {}
    hotspot_profile_deltas = payload.get("hotspot_profile_deltas") or {}
    observations: list[str] = []
    seen_observations: set[str] = set()

    def add_observation(text: str) -> None:
        if text in seen_observations:
            return
        seen_observations.add(text)
        observations.append(text)

    occ_item = runtime_deltas.get("occupancy_average_active") or {}
    stall_item = runtime_deltas.get("avg_stall_per_inst") or {}
    duration_item = hotspot_profile_deltas.get("avg_duration_per_hit") or {}
    occ_delta = occ_item.get("delta")
    stall_delta = stall_item.get("delta")
    duration_delta = duration_item.get("delta")
    occ_ratio = occ_item.get("delta_ratio")
    stall_ratio = stall_item.get("delta_ratio")
    duration_ratio = duration_item.get("delta_ratio")
    if isinstance(occ_delta, (int, float)) and isinstance(occ_ratio, (int, float)) and occ_ratio > 0.05:
        add_observation(f"occupancy_average_active: +{_fmt_float(occ_delta)} ({_fmt_ratio_pct(occ_ratio)})")
    if isinstance(stall_delta, (int, float)) and isinstance(stall_ratio, (int, float)) and stall_ratio < -0.05:
        add_observation(
            f"avg_stall_per_inst: {_fmt_float(stall_item.get('before'))} -> {_fmt_float(stall_item.get('after'))} "
            f"(delta={_fmt_float(stall_delta)}, ratio={_fmt_ratio_pct(stall_ratio)})"
        )
    if isinstance(duration_delta, (int, float)) and isinstance(duration_ratio, (int, float)) and duration_ratio < -0.05:
        add_observation(
            f"hotspot_avg_duration_per_hit: {_fmt_float(duration_item.get('before'))} -> {_fmt_float(duration_item.get('after'))} "
            f"(delta={_fmt_float(duration_delta)}, ratio={_fmt_ratio_pct(duration_ratio)})"
        )
    if isinstance(occ_delta, (int, float)) and isinstance(occ_ratio, (int, float)) and occ_ratio < -0.05:
        add_observation(
            f"occupancy_average_active: {_fmt_float(occ_item.get('before'))} -> {_fmt_float(occ_item.get('after'))} "
            f"(delta={_fmt_float(occ_delta)}, ratio={_fmt_ratio_pct(occ_ratio)})"
        )
    if isinstance(stall_delta, (int, float)) and isinstance(stall_ratio, (int, float)) and stall_ratio > 0.05:
        add_observation(
            f"avg_stall_per_inst: {_fmt_float(stall_item.get('before'))} -> {_fmt_float(stall_item.get('after'))} "
            f"(delta=+{_fmt_float(stall_delta)}, ratio=+{_fmt_ratio_pct(stall_ratio)})"
        )
    if isinstance(duration_delta, (int, float)) and isinstance(duration_ratio, (int, float)) and duration_ratio > 0.05:
        add_observation(
            f"hotspot_avg_duration_per_hit: {_fmt_float(duration_item.get('before'))} -> {_fmt_float(duration_item.get('after'))} "
            f"(delta=+{_fmt_float(duration_delta)}, ratio=+{_fmt_ratio_pct(duration_ratio)})"
        )
    for proxy_name in (
        "sync_wait_share", "sync_wait_cycles_per_inst", "immed_stall_per_inst", "lds_stall_per_inst",
        "global_memory_duration_share", "global_memory_stall_share", "lds_duration_share", "lds_stall_share",
    ):
        item = runtime_proxy_deltas.get(proxy_name) or {}
        delta = item.get("delta")
        ratio = item.get("delta_ratio")
        threshold = 0.05 if "share" in proxy_name else 1.0
        if isinstance(delta, (int, float)) and abs(delta) >= threshold:
            sign = "+" if delta > 0 else ""
            ratio_text = ""
            if isinstance(ratio, (int, float)):
                ratio_sign = "+" if ratio > 0 else ""
                ratio_text = f", ratio={ratio_sign}{_fmt_ratio_pct(ratio)}"
            add_observation(
                f"{proxy_name}: {_fmt_float(item.get('before'))} -> {_fmt_float(item.get('after'))} "
                f"(delta={sign}{_fmt_float(delta)}{ratio_text})"
            )
            if len(observations) >= 5:
                break
    for category_name in ("VALU", "SALU", "LDS", "VMEM", "SMEM"):
        item = category_deltas.get(category_name) or {}
        delta = item.get("delta")
        ratio = item.get("delta_ratio")
        if isinstance(delta, (int, float)) and isinstance(ratio, (int, float)) and abs(ratio) > 0.15 and abs(delta) >= 8:
            sign = "+" if delta > 0 else ""
            ratio_sign = "+" if ratio > 0 else ""
            add_observation(
                f"{category_name}: {item.get('before')} -> {item.get('after')} "
                f"(delta={sign}{delta}, ratio={ratio_sign}{_fmt_ratio_pct(ratio)})"
            )
            if len(observations) >= 5:
                break
    if observations:
        lines.append("  observations:")
        for item in observations[:5]:
            lines.append(f"    - {item}")
    return "\n".join(lines)
