from __future__ import annotations

from typing import Any


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


def render_code_object_isa_payload(payload: dict[str, Any]) -> str:
    lines = ["code_object_isa:"]
    lines.append(f"  file: {payload.get('file')}")
    lines.append(f"  focus_code_object: {payload.get('focus_code_object_index')}")
    lines.append(f"  entry_point: {payload.get('entry_point')}")
    lines.append(f"  symbol: {payload.get('symbol')}")
    source_hints = payload.get("source_hints") or {}
    if source_hints:
        lines.append(
            "  source_hints: "
            f"file={source_hints.get('file')} available={source_hints.get('available')} "
            f"match_count={source_hints.get('match_count')}"
        )
        for item in (source_hints.get("matches") or [])[:8]:
            lines.append(f"    - line={item.get('line')} match={item.get('match')}")
    source_isa_blocks = payload.get("source_isa_blocks") or []
    if source_isa_blocks:
        lines.append("  source_isa_blocks:")
        for block in source_isa_blocks[:8]:
            lines.append(f"    - label={block.get('label')}")
            for src in (block.get("source_lines") or [])[:4]:
                lines.append(f"      source line={src.get('line')} match={src.get('match')}")
            for ins in (block.get("isa_instructions") or [])[:6]:
                line = (
                    f"      isa pc=0x{int(ins.get('pc', 0) or 0):x} "
                    f"{ins.get('mnemonic')} {ins.get('operands')}"
                )
                if ins.get("text"):
                    line += f" isa=\"{ins.get('text')}\""
                lines.append(line)
    top_pcs = payload.get("top_pcs") or []
    if top_pcs:
        lines.append("  top_pcs:")
        for item in top_pcs[:8]:
            line = (
                f"    - pc=0x{int(item.get('pc', 0) or 0):x} "
                f"{item.get('mnemonic')} {item.get('operands')}"
            )
            if item.get("text"):
                line += f" isa=\"{item.get('text')}\""
            lines.append(line)
    instructions = payload.get("instructions") or []
    if instructions:
        lines.append("  isa:")
        for item in instructions:
            line = (
                f"    - pc=0x{int(item.get('pc', 0) or 0):x} "
                f"{item.get('mnemonic')} {item.get('operands')}"
            )
            if item.get("text"):
                line += f" isa=\"{item.get('text')}\""
            lines.append(line)
    return "\n".join(lines)


def render_shader_focus_payload(payload: dict[str, Any], *, source_excerpt: bool = False) -> str:
    lines = ["shader_focus:"]
    lines.append(f"  file: {payload.get('file')}")
    lines.append(f"  focus_code_object: {payload.get('focus_code_object_index')}")
    lines.append(f"  enough_for_shader_tuning: {payload.get('enough_for_shader_tuning')}")
    lines.append(
        "  runtime_scope: "
        f"{payload.get('runtime_scope')} focused_runtime_available={payload.get('focused_runtime_available')}"
    )
    trace_quality = payload.get("trace_quality") or {}
    lines.append(
        "  trace_quality: "
        f"level={trace_quality.get('runtime_evidence_level')} "
        f"sqtt_bytes={trace_quality.get('sqtt_trace_bytes')} "
        f"dispatch_spans={trace_quality.get('dispatch_span_count')} "
        f"mapped_dispatch={trace_quality.get('mapped_dispatch_count')}/{trace_quality.get('total_dispatch_count')}"
    )
    profiling_constraints = payload.get("profiling_constraints") or {}
    if profiling_constraints:
        lines.append(
            "  profiling_constraints: "
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
    catalog = payload.get("code_object_catalog") or []
    if len(catalog) > 1:
        lines.append("  code_objects:")
        for item in catalog[:6]:
            lines.append(
                f"    - code_object[{item.get('index')}] entry_point={item.get('entry_point')} "
                f"vgpr={item.get('vgpr_count')} sgpr={item.get('sgpr_count')} "
                f"lds={item.get('lds_size')} scratch={item.get('scratch_memory_size')} "
                f"dispatch_assignments={item.get('dispatch_assignment_count')}"
            )
    runtime = payload.get("runtime") or {}
    runtime_hotspots = payload.get("runtime_hotspots") or []
    if runtime:
        cats = runtime.get("category_counts") or {}
        lines.append(
            "  runtime(global): "
            f"instructions={runtime.get('instructions')} "
            f"VALU={cats.get('VALU', 0)} SALU={cats.get('SALU', 0)} "
            f"LDS={cats.get('LDS', 0)} VMEM={cats.get('VMEM', 0)} SMEM={cats.get('SMEM', 0)} "
            f"avg_stall={_fmt_float(runtime.get('avg_stall_per_inst'))} "
            f"stalled_inst_share={_fmt_float(runtime.get('stalled_instruction_share'))} "
            f"avg_wave_lifetime={_fmt_float(runtime.get('avg_wave_lifetime'))} "
            f"stall_share={_fmt_float(runtime.get('stall_share_of_duration'))} "
            f"occ_avg={_fmt_float(runtime.get('occupancy_average_active'))} "
            f"occ_max={runtime.get('occupancy_max_active')}"
        )
        top_hotspot = runtime.get("top_hotspot_profile") or {}
        if top_hotspot:
            symbol = top_hotspot.get("symbol") or {}
            symbol_text = ""
            if symbol.get("name"):
                symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
            bucket_text = ""
            if int(top_hotspot.get("address", 0) or 0) == 0:
                bucket_text = " bucket=kernel_entry"
            lines.append(
                "  runtime_top_hotspot: "
                f"address=0x{int(top_hotspot.get('address', 0) or 0):x} "
                f"duration={top_hotspot.get('total_duration')} stall={top_hotspot.get('total_stall')} "
                f"hitcount={top_hotspot.get('hitcount')} "
                f"avg_duration={_fmt_float(top_hotspot.get('avg_duration_per_hit'))} "
                f"avg_stall={_fmt_float(top_hotspot.get('avg_stall_per_hit'))}{symbol_text}{bucket_text}"
            )
    runtime_proxies = payload.get("runtime_proxies") or {}
    if runtime_proxies:
        lines.append(
            "  runtime_proxies: "
            f"mem_inst_share={_fmt_float(runtime_proxies.get('memory_instruction_share'))} "
            f"mem_duration_share={_fmt_float(runtime_proxies.get('memory_duration_share'))} "
            f"mem_stall_share={_fmt_float(runtime_proxies.get('memory_stall_share'))} "
            f"global_mem_duration_share={_fmt_float(runtime_proxies.get('global_memory_duration_share'))} "
            f"global_mem_stall_share={_fmt_float(runtime_proxies.get('global_memory_stall_share'))} "
            f"lds_duration_share={_fmt_float(runtime_proxies.get('lds_duration_share'))} "
            f"lds_stall_share={_fmt_float(runtime_proxies.get('lds_stall_share'))} "
            f"scalar_duration_share={_fmt_float(runtime_proxies.get('scalar_duration_share'))} "
            f"scalar_stall_share={_fmt_float(runtime_proxies.get('scalar_stall_share'))} "
            f"vector_duration_share={_fmt_float(runtime_proxies.get('vector_duration_share'))} "
            f"vector_stall_share={_fmt_float(runtime_proxies.get('vector_stall_share'))} "
            f"sync_wait_share={_fmt_float(runtime_proxies.get('sync_wait_share'))} "
            f"sync_wait_per_inst={_fmt_float(runtime_proxies.get('sync_wait_cycles_per_inst'))} "
            f"immed_stall_per_inst={_fmt_float(runtime_proxies.get('immed_stall_per_inst'))} "
            f"lds_stall_per_inst={_fmt_float(runtime_proxies.get('lds_stall_per_inst'))}"
        )
    category_profiles = payload.get("runtime_category_profiles") or []
    if category_profiles:
        lines.append("  runtime_category_ranking:")
        for item in category_profiles[:5]:
            lines.append(
                "    - "
                f"{item.get('category')} count={item.get('count')} "
                f"count_share={_fmt_float(item.get('count_share'))} "
                f"duration={item.get('duration_total')} stall={item.get('stall_total')} "
                f"duration_share={_fmt_float(item.get('duration_share'))} "
                f"stall_share={_fmt_float(item.get('stall_share'))} "
                f"stall_over_duration={_fmt_float(item.get('stall_share_of_duration'))}"
            )
    wave_state_profiles = payload.get("wave_state_profiles") or []
    if wave_state_profiles:
        lines.append("  wave_state_ranking:")
        for item in wave_state_profiles[:5]:
            lines.append(
                "    - "
                f"{item.get('state')} duration={item.get('duration')} "
                f"share={_fmt_float(item.get('share'))}"
            )
    if len(runtime_hotspots) > 1:
        lines.append("  runtime_hotspot_ranking:")
        for item in runtime_hotspots[:4]:
            symbol = item.get("symbol") or {}
            symbol_text = ""
            if symbol.get("name"):
                symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
            lines.append(
                "    - "
                f"address=0x{int(item.get('address', 0) or 0):x} "
                f"duration={item.get('total_duration')} stall={item.get('total_stall')} "
                f"hitcount={item.get('hitcount')} "
                f"avg_duration={_fmt_float(item.get('avg_duration_per_hit'))} "
                f"avg_stall={_fmt_float(item.get('avg_stall_per_hit'))}{symbol_text}"
            )
    hotspot_candidates = payload.get("runtime_hotspot_candidates") or []
    if hotspot_candidates:
        lines.append("  focused_runtime_hotspots:")
        for item in hotspot_candidates[:4]:
            symbol = item.get("symbol") or {}
            symbol_text = ""
            if symbol.get("name"):
                symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
            bucket_text = ""
            if int(item.get("address", 0) or 0) == 0:
                bucket_text = " bucket=kernel_entry"
            lines.append(
                "    - "
                f"address=0x{int(item.get('address', 0) or 0):x} duration={item.get('total_duration')} "
                f"stall={item.get('total_stall')} hitcount={item.get('hitcount')} "
                f"avg_duration={_fmt_float(item.get('avg_duration_per_hit'))} "
                f"avg_stall={_fmt_float(item.get('avg_stall_per_hit'))} "
                f"dispatch_share={_fmt_float(item.get('dispatch_assignment_share'))} "
                f"dispatch_isa_share={_fmt_float(item.get('dispatch_isa_mapped_dispatch_share'))} "
                f"match_kind={item.get('match_kind')}{symbol_text}{bucket_text}"
            )
            for pc in (item.get("top_pcs") or [])[:3]:
                line = (
                    "      "
                    f"pc=0x{int(pc.get('pc', 0) or 0):x} {pc.get('mnemonic')} {pc.get('operands')} "
                    f"category={pc.get('category')} count={pc.get('count')}"
                )
                if pc.get("text"):
                    line += f" isa=\"{pc.get('text')}\""
                lines.append(line)
    dispatch_isa = payload.get("dispatch_isa") or {}
    lines.append(
        "  dispatch_isa: "
        f"mapped={dispatch_isa.get('mapped_dispatch_count')}/{dispatch_isa.get('dispatch_count')}"
    )
    top_pcs = dispatch_isa.get("top_pcs") or []
    if top_pcs:
        lines.append("  top_pcs:")
        for item in top_pcs[:4]:
            line = (
                f"    - code_object[{item.get('code_object_index')}]"
                f":0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic')} {item.get('operands')} "
                f"category={item.get('category')} count={item.get('count')}"
            )
            if item.get("text"):
                line += f" isa=\"{item.get('text')}\""
            lines.append(line)
    instruction_ranking = payload.get("instruction_ranking") or []
    if instruction_ranking:
        lines.append("  instruction_ranking:")
        for item in instruction_ranking[:6]:
            line = (
                f"    - pc=0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic')} {item.get('operands')} "
                f"category={item.get('category')} dispatch_count={item.get('dispatch_count')} "
                f"hotspot_mentions={item.get('hotspot_mentions')} score={item.get('score')}"
            )
            if item.get("text"):
                line += f" isa=\"{item.get('text')}\""
            lines.append(line)
    source_regions = payload.get("source_region_metrics") or []
    if source_regions:
        lines.append("  source_regions:")
        for item in source_regions[:6]:
            lines.append(
                "    - "
                f"label={item.get('label')} "
                f"attribution_mode={item.get('attribution_mode')} "
                f"dispatch_count={item.get('dispatch_count')} "
                f"score={item.get('score')} "
                f"hotspot_duration={item.get('hotspot_duration')} "
                f"hotspot_stall={item.get('hotspot_stall')} "
                f"avg_hotspot_stall={_fmt_float(item.get('avg_hotspot_stall'))} "
                f"proxy_score={_fmt_float(item.get('proxy_score'))}"
            )
            for metric_item in (item.get("proxy_metric_rows") or [])[:3]:
                lines.append(
                    "      "
                    f"proxy_metric={metric_item.get('metric')} "
                    f"value={_fmt_float(metric_item.get('value'))} "
                    f"weight={_fmt_float(metric_item.get('weight'))} "
                    f"weighted={_fmt_float(metric_item.get('weighted_value'))}"
                )
            for line_no in (item.get("source_lines") or [])[:4]:
                lines.append(f"      source_line={line_no}")
            for pc in (item.get("top_pcs") or [])[:4]:
                lines.append(f"      hot_pc={pc}")
    memory_access_hints = payload.get("memory_access_hints") or {}
    if memory_access_hints:
        proxies = memory_access_hints.get("proxies") or {}
        lines.append(
            "  memory_access_hints: "
            f"global_mem_duration_share={_fmt_float(proxies.get('global_memory_duration_share'))} "
            f"global_mem_stall_share={_fmt_float(proxies.get('global_memory_stall_share'))} "
            f"lds_duration_share={_fmt_float(proxies.get('lds_duration_share'))} "
            f"lds_stall_share={_fmt_float(proxies.get('lds_stall_share'))} "
            f"lds_stall_per_inst={_fmt_float(proxies.get('lds_stall_per_inst'))}"
        )
        for item in (memory_access_hints.get("source_matches") or [])[:4]:
            lines.append(f"    - line={item.get('line')} match={item.get('match')}")
    bottleneck_hints = payload.get("bottleneck_hints") or []
    if bottleneck_hints:
        lines.append("  bottleneck_metrics:")
        for item in bottleneck_hints[:5]:
            lines.append(
                "    - "
                f"metric={item.get('name')} value={_fmt_float(item.get('value'))} "
                f"threshold={_fmt_float(item.get('threshold'))}"
            )
    occupancy_detail = payload.get("occupancy_detail") or {}
    if occupancy_detail:
        lines.append(
            "  occupancy_detail: "
            f"runtime_average_active={_fmt_float(occupancy_detail.get('runtime_average_active'))} "
            f"runtime_max_active={occupancy_detail.get('runtime_max_active')} "
            f"avg_wave_lifetime={_fmt_float(occupancy_detail.get('avg_wave_lifetime'))} "
            f"max_wave_lifetime={occupancy_detail.get('max_wave_lifetime')} "
            f"stalled_instruction_share={_fmt_float(occupancy_detail.get('stalled_instruction_share'))} "
            f"vgpr={occupancy_detail.get('vgpr_count')} "
            f"lds={occupancy_detail.get('lds_size')} "
            f"wavefront={occupancy_detail.get('wavefront_size')}"
        )
    event_barrier_context = payload.get("event_barrier_context") or {}
    if event_barrier_context:
        lines.append(
            "  event_barrier_context: "
            f"dispatch_spans={event_barrier_context.get('dispatch_api_span_count')} "
            f"dispatch_assignments={event_barrier_context.get('dispatch_span_assignment_count')} "
            f"bind_markers={event_barrier_context.get('bind_marker_count')} "
            f"cb_spans={event_barrier_context.get('command_buffer_span_count')} "
            f"barrier_markers={event_barrier_context.get('barrier_marker_count')} "
            f"barrier_spans={event_barrier_context.get('barrier_span_count')} "
            f"unmatched_barrier_begins={event_barrier_context.get('unmatched_barrier_begin_count')} "
            f"dispatches_per_cb={_fmt_float(event_barrier_context.get('dispatches_per_cb'))} "
            f"barriers_per_dispatch={_fmt_float(event_barrier_context.get('barriers_per_dispatch'))}"
        )
    tuning_summary = payload.get("tuning_summary") or {}
    if tuning_summary:
        lines.append(
            "  tuning_summary: "
            f"dispatches_per_cb={_fmt_float(tuning_summary.get('dispatches_per_cb'))} "
            f"barriers_per_dispatch={_fmt_float(tuning_summary.get('barriers_per_dispatch'))}"
        )
        metric_rows = tuning_summary.get("metric_rows") or []
        for item in metric_rows[:4]:
            lines.append(f"    - {_fmt_metric_row(item)}")
        hot_pcs = tuning_summary.get("hot_pcs") or []
        for pc in hot_pcs[:4]:
            lines.append(f"    - hot_pc={pc}")
        source_lines = tuning_summary.get("source_lines") or []
        for line in source_lines[:6]:
            lines.append(f"    - source_line={line}")
    source_hints = payload.get("source_hints") or {}
    if source_hints:
        lines.append(
            "  source_hints: "
            f"file={source_hints.get('file')} available={source_hints.get('available')} "
            f"match_count={source_hints.get('match_count')}"
        )
        for item in (source_hints.get("matches") or [])[:6]:
            lines.append(f"    - line={item.get('line')} match={item.get('match')}")
            if source_excerpt:
                for excerpt in (item.get("excerpt") or [])[:3]:
                    prefix = ">" if excerpt.get("focus") else " "
                    lines.append(f"      {prefix}L{excerpt.get('line')}: {excerpt.get('text')}")
    capabilities = payload.get("capture_capabilities") or {}
    if capabilities:
        lines.append(
            "  capture_capabilities: "
            f"trace_path={capabilities.get('trace_path')} "
            f"dynamic_instruction_timing={capabilities.get('has_dynamic_instruction_timing')} "
            f"dynamic_instruction_count={capabilities.get('has_dynamic_instruction_count')} "
            f"active_lane_count={capabilities.get('has_active_lane_count')} "
            f"not_issued_reason={capabilities.get('has_not_issued_reason')} "
            f"memory_stride={capabilities.get('has_memory_access_stride')} "
            f"cacheline_efficiency={capabilities.get('has_cacheline_efficiency')}"
        )
    return "\n".join(lines)
