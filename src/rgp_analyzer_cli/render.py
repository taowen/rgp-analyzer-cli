from __future__ import annotations

from typing import Any

from .legacy_render import render_inspect as render_v1_inspect
from .legacy_render import render_stitch_report as render_v1_stitch_report
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
            lines.append(
                f"    - line={item.get('line')} match={item.get('match')}"
            )
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
            lines.append(
                f"    - line={item.get('line')} match={item.get('match')}"
            )
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


def render_compare_shader_focus_payload(payload: dict[str, Any], *, source_excerpt: bool = False) -> str:
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
            lines.append(
                f"    - line={item.get('line')} match={item.get('match')}"
            )
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
                parts.append(
                    f"duration {duration.get('before')} -> {duration.get('after')} (delta={duration.get('delta')})"
                )
            if stall:
                parts.append(
                    f"stall {stall.get('before')} -> {stall.get('after')} (delta={stall.get('delta')})"
                )
            if avg_duration:
                parts.append(
                    f"avg_duration { _fmt_float(avg_duration.get('before')) } -> { _fmt_float(avg_duration.get('after')) }"
                )
            if avg_stall:
                parts.append(
                    f"avg_stall { _fmt_float(avg_stall.get('before')) } -> { _fmt_float(avg_stall.get('after')) }"
                )
            lines.append(line + " " + "; ".join(parts))
            top_pcs = item.get("top_pcs") or []
            for pc in top_pcs[:3]:
                lines.append(
                    f"      pc_delta {pc.get('signature')}: {pc.get('before')} -> {pc.get('after')} "
                    f"(delta={pc.get('delta')})"
                )
    runtime_deltas = payload.get("runtime_deltas") or {}
    category_deltas = payload.get("runtime_category_count_deltas") or {}
    category_duration_deltas = payload.get("runtime_category_duration_deltas") or {}
    category_stall_deltas = payload.get("runtime_category_stall_deltas") or {}
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
        add_observation(
            f"occupancy_average_active: +{_fmt_float(occ_delta)} ({_fmt_ratio_pct(occ_ratio)})"
        )
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
        "sync_wait_share",
        "sync_wait_cycles_per_inst",
        "immed_stall_per_inst",
        "lds_stall_per_inst",
        "global_memory_duration_share",
        "global_memory_stall_share",
        "lds_duration_share",
        "lds_stall_share",
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
    for category_name in ("VALU", "SALU", "LDS", "VMEM", "SMEM"):
        item = category_duration_deltas.get(category_name) or {}
        delta = item.get("delta")
        ratio = item.get("delta_ratio")
        if isinstance(delta, (int, float)) and isinstance(ratio, (int, float)) and abs(ratio) > 0.15 and abs(delta) >= 200:
            sign = "+" if delta > 0 else ""
            ratio_sign = "+" if ratio > 0 else ""
            add_observation(
                f"{category_name}_duration: {item.get('before')} -> {item.get('after')} "
                f"(delta={sign}{delta}, ratio={ratio_sign}{_fmt_ratio_pct(ratio)})"
            )
            if len(observations) >= 5:
                break
    for category_name in ("VALU", "SALU", "LDS", "VMEM", "SMEM"):
        item = category_stall_deltas.get(category_name) or {}
        delta = item.get("delta")
        ratio = item.get("delta_ratio")
        if isinstance(delta, (int, float)) and isinstance(ratio, (int, float)) and abs(ratio) > 0.15 and abs(delta) >= 100:
            sign = "+" if delta > 0 else ""
            ratio_sign = "+" if ratio > 0 else ""
            add_observation(
                f"{category_name}_stall: {item.get('before')} -> {item.get('after')} "
                f"(delta={sign}{delta}, ratio={ratio_sign}{_fmt_ratio_pct(ratio)})"
            )
            if len(observations) >= 5:
                break
    for proxy_name in (
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
    if observations:
        lines.append("  observations:")
        for item in observations[:5]:
            lines.append(f"    - {item}")
    return "\n".join(lines)
