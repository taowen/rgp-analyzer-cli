#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("captures:") or stripped.startswith("phase_markers:"):
            continue
        if stripped.startswith("-") or stripped.startswith("captures"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.startswith("phase_") or key.startswith("  "):
            continue
        result[key.strip()] = value.strip()
    return result


def _fmt_float(value: object, digits: int = 2) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    if len(sys.argv) not in {2, 3, 4}:
        print(
            f"usage: {Path(sys.argv[0]).name} <triage-json> [<perf-json>] [<manifest>]",
            file=sys.stderr,
        )
        return 2

    triage = _load_json(Path(sys.argv[1]))
    perf_rows = []
    perf_families = []
    perf_total_time_us = None
    manifest: dict[str, str] = {}
    if len(sys.argv) >= 3:
        third = Path(sys.argv[2])
        if third.suffix == ".txt":
            manifest = _load_manifest(third)
        else:
            perf_payload = _load_json(third)
            if isinstance(perf_payload, dict):
                perf_total_time_us = perf_payload.get("total_time_us")
                perf_rows = perf_payload.get("rows") or []
                perf_families = perf_payload.get("families") or []
            else:
                perf_rows = perf_payload
    if len(sys.argv) == 4:
        manifest = _load_manifest(Path(sys.argv[3]))

    summary = triage.get("summary") or {}
    trace_quality = summary.get("trace_quality") or {}
    profiling_constraints = summary.get("profiling_constraints") or {}
    resource = summary.get("resource") or {}
    runtime = summary.get("runtime") or {}
    top_category = runtime.get("top_category") or {}
    top_hotspot = runtime.get("top_hotspot_profile") or {}

    print("workload_evidence:")
    if manifest:
        print(
            "  capture_profile: "
            f"profile_mode={manifest.get('profile_mode')} "
            f"stop_after_phase={manifest.get('stop_after_phase')} "
            f"trace_mode={manifest.get('trace_mode')} "
            f"app_status={manifest.get('app_status')}"
        )
        print(
            "  capture_submit_policy: "
            f"nodes_per_submit={manifest.get('nodes_per_submit')} "
            f"matmul_bytes_per_submit={manifest.get('matmul_bytes_per_submit')} "
            f"disable_fusion={manifest.get('disable_fusion')} "
            f"disable_graph_optimize={manifest.get('disable_graph_optimize')}"
        )
    if perf_rows:
        top_perf = perf_rows[0]
        if perf_total_time_us is not None:
            print(f"  ggml_perf_total: total_us={float(perf_total_time_us):.1f}")
        print(
            "  ggml_perf_top_op: "
            f"total_us={float(top_perf.get('total_us', 0)):.1f} "
            f"count={int(top_perf.get('count', 0))} "
            f"name={top_perf.get('name')}"
        )
    if perf_families:
        top_family = perf_families[0]
        print(
            "  ggml_perf_top_family: "
            f"total_us={float(top_family.get('total_us', 0)):.1f} "
            f"count={int(top_family.get('count', 0))} "
            f"variants={int(top_family.get('variants', 0))} "
            f"family={top_family.get('family')}"
        )
    print(
        "  rgp_trace_quality: "
        f"level={trace_quality.get('runtime_evidence_level')} "
        f"sqtt_bytes={trace_quality.get('sqtt_trace_bytes')} "
        f"queue_events={trace_quality.get('queue_event_count')} "
        f"instructions={trace_quality.get('decoded_instruction_count')} "
        f"dispatch_spans={trace_quality.get('dispatch_span_count')} "
        f"mapped_dispatch={trace_quality.get('mapped_dispatch_count')}/{trace_quality.get('total_dispatch_count')}"
    )
    if profiling_constraints:
        print(
            "  profiling_constraints: "
            f"submit_dilution_suspected={profiling_constraints.get('submit_dilution_suspected')} "
            f"reason={profiling_constraints.get('reason')} "
            f"sparse_runtime_trace={profiling_constraints.get('sparse_runtime_trace')}"
        )
    if resource:
        print(
            "  shader_resource: "
            f"entry_point={resource.get('entry_point')} "
            f"vgpr={resource.get('vgpr_count')} "
            f"sgpr={resource.get('sgpr_count')} "
            f"lds={resource.get('lds_size')} "
            f"scratch={resource.get('scratch_memory_size')}"
        )
    if top_category:
        print(
            "  runtime_top_category: "
            f"category={top_category.get('category')} "
            f"count={top_category.get('count')} "
            f"duration={top_category.get('duration_total')} "
            f"stall={top_category.get('stall_total')} "
            f"stall_share={_fmt_float(top_category.get('stall_share_of_duration'))}"
        )
    if top_hotspot:
        symbol = top_hotspot.get("symbol") or {}
        symbol_text = ""
        if symbol.get("name"):
            symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
        print(
            "  runtime_top_hotspot: "
            f"address=0x{int(top_hotspot.get('address', 0) or 0):x} "
            f"duration={top_hotspot.get('total_duration')} "
            f"stall={top_hotspot.get('total_stall')} "
            f"hitcount={top_hotspot.get('hitcount')}{symbol_text}"
        )

    decoder = summary.get("decoder") or {}
    if decoder:
        print(
            "  decoder_status: "
            f"status={decoder.get('status')} "
            f"load_failures={decoder.get('code_object_load_failures')} "
            f"dispatch_isa={decoder.get('dispatch_isa_mapped')}/{decoder.get('dispatch_isa_total')} "
            f"sparse_runtime_trace={decoder.get('sparse_runtime_trace')}"
        )

    stitch = summary.get("stitch") or {}
    if stitch:
        print(
            "  stitch_context: "
            f"resolved={stitch.get('resolved_entry_count')} "
            f"dispatch_spans={stitch.get('dispatch_api_span_count')} "
            f"assignments={stitch.get('dispatch_span_assignment_count')} "
            f"bind_markers={stitch.get('bind_marker_count')} "
            f"barrier_spans={stitch.get('barrier_span_count')}"
        )

    observations = triage.get("findings") or []
    if observations:
        print("  observations:")
        for item in observations[:3]:
            print(f"    - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
