from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .decode_bridge import build_stitch_model, collect_code_object_records
from .native_decode import default_decoder_lib_dir, repo_root as decode_repo_root, resolve_helper_path, run_decode_helper
from .parser import materialize_code_object_payload
from .resource_metadata import extract_resource_metadata
from .analyze import summarize_isa_text
from .marker_scan import scan_general_api_markers, scan_markers, summarize_general_api_markers, summarize_markers
from .tinygrad_support.isa_map import map_dispatch_spans_to_isa, map_tinygrad_packets_to_isa


def summarize_code_objects_isa(report: dict[str, Any], tool: str | None = None, limit: int = 10, symbol: str | None = None) -> list[dict[str, Any]]:
    objdump = shutil.which(tool) if tool else None
    objdump = objdump or shutil.which("llvm-objdump-18") or shutil.which("llvm-objdump")
    if objdump is None:
        raise RuntimeError("llvm-objdump tool not found")

    records = collect_code_object_records(report)[:limit]
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rgp-analyzer-triage-isa-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for record in records:
            payload = materialize_code_object_payload(report, record, pad_elf=True)
            path = tmpdir_path / f"code-object-{record['index']:03d}.elf"
            path.write_bytes(payload)
            cmd = [objdump, "-d", str(path)]
            if symbol:
                cmd = [objdump, "-d", f"--disassemble-symbols={symbol}", str(path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            results.append(
                {
                    "index": record["index"],
                    "returncode": proc.returncode,
                    "tool": objdump,
                    "summary": summarize_isa_text(proc.stdout),
                }
            )
    return results


def _first_stream_summary(decode_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decode_json:
        return None
    stitched = decode_json.get("stitched") or {}
    streams = stitched.get("streams") or decode_json.get("streams") or []
    return streams[0] if streams else None


def _triage_signals(
    report: dict[str, Any],
    *,
    helper: Path,
    decoder_lib_dir: Path,
    isa_tool: str | None,
    readelf_tool: str | None,
    limit: int,
    hotspot_limit: int,
) -> dict[str, Any]:
    decode_result = run_decode_helper(
        report,
        helper=helper,
        decoder_lib_dir=decoder_lib_dir,
        stream_limit=1,
        hotspot_limit=hotspot_limit,
        strict=False,
        as_json=True,
        keep_temp=False,
    )
    decode_json = decode_result.get("json")
    marker_result = scan_general_api_markers(report, stream_limit=1)
    raw_marker_result = scan_markers(report, stream_limit=1)
    try:
        tinygrad_map = map_tinygrad_packets_to_isa(report, tool=isa_tool, stream_index=0, packet_limit=5000, mapped_limit=64)
    except Exception:
        tinygrad_map = None
    try:
        dispatch_isa_map = map_dispatch_spans_to_isa(
            report, tool=isa_tool, stream_index=0, dispatch_limit=6, context_packets=64, packet_limit=5000, mapped_limit=24
        )
    except Exception:
        dispatch_isa_map = None
    return {
        "runtime_signals": {
            "decode": decode_json,
            "decode_stream": _first_stream_summary(decode_json),
        },
        "stitch_signals": build_stitch_model(report),
        "marker_signals": {
            "general_api": summarize_general_api_markers(marker_result),
            "raw_markers": summarize_markers(raw_marker_result),
        },
        "static_signals": {
            "resource_summary": extract_resource_metadata(report, tool=readelf_tool, limit=limit),
            "isa_summary": summarize_code_objects_isa(report, tool=isa_tool, limit=limit, symbol="_amdgpu_cs_main"),
        },
        "experimental_signals": {
            "tinygrad_isa_map": tinygrad_map,
            "dispatch_isa_map": dispatch_isa_map,
        },
    }


def _triage_findings(
    resource: dict[str, Any] | None,
    isa: dict[str, Any] | None,
    decode_json: dict[str, Any] | None,
    decode_stream: dict[str, Any] | None,
    marker_summary: dict[str, Any] | None,
    marker_totals: dict[str, Any] | None,
    stitch_model: dict[str, Any] | None,
    tinygrad_map: dict[str, Any] | None,
    dispatch_isa_map: dict[str, Any] | None,
) -> list[str]:
    findings: list[str] = []

    if resource:
        vgpr = resource.get("vgpr_count")
        sgpr = resource.get("sgpr_count")
        scratch = resource.get("scratch_memory_size")
        lds = resource.get("lds_size")
        wavefront = resource.get("wavefront_size")
        if isinstance(vgpr, int) and vgpr >= 96:
            findings.append(f"High VGPR pressure candidate: vgpr_count={vgpr}.")
        elif isinstance(vgpr, int) and vgpr >= 64:
            findings.append(f"Moderate VGPR pressure candidate: vgpr_count={vgpr}.")
        if isinstance(sgpr, int) and sgpr >= 128:
            findings.append(f"High SGPR usage candidate: sgpr_count={sgpr}.")
        if isinstance(scratch, int) and scratch > 0:
            findings.append(f"Scratch/spill risk is explicit in metadata: scratch_memory_size={scratch}.")
        if isinstance(lds, int) and lds > 0:
            findings.append(f"LDS usage is explicit in metadata: lds_size={lds}.")
        if wavefront == 64:
            findings.append("Wave64 kernel metadata detected.")

    if isa:
        summary = isa.get("summary", {})
        inst = summary.get("instruction_count", 0)
        valu = summary.get("vector_alu", 0)
        salu = summary.get("scalar_ops", 0)
        lds_ops = summary.get("lds_ops", 0)
        buffer_ops = summary.get("buffer_ops", 0)
        flat_ops = summary.get("flat_ops", 0)
        if inst:
            if valu / inst >= 0.7:
                findings.append(f"ISA looks VALU-heavy: vector_alu={valu}/{inst}.")
            if (buffer_ops + flat_ops) / inst >= 0.15:
                findings.append(f"ISA shows notable memory traffic: buffer_ops+flat_ops={buffer_ops + flat_ops}/{inst}.")
            if lds_ops > 0:
                findings.append(f"ISA includes LDS instructions: lds_ops={lds_ops}.")
            if salu / inst >= 0.2:
                findings.append(f"ISA has a non-trivial scalar control component: scalar_ops={salu}/{inst}.")

    if decode_stream:
        instructions = decode_stream.get("instructions", 0)
        cats = decode_stream.get("category_counts", {})
        valu = cats.get("VALU", 0)
        vmem = cats.get("VMEM", 0)
        smem = cats.get("SMEM", 0)
        lds = cats.get("LDS", 0)
        if instructions:
            if valu / instructions >= 0.85:
                findings.append(f"Runtime trace is overwhelmingly VALU-bound: VALU={valu}/{instructions}.")
            if (vmem + smem) / instructions >= 0.05:
                findings.append(f"Runtime trace includes visible memory operations: VMEM+SMEM={vmem + smem}/{instructions}.")
            if lds > 0:
                findings.append(f"Runtime trace contains LDS operations: LDS={lds}/{instructions}.")
        hotspots = decode_stream.get("annotated_hotspots") or decode_stream.get("hotspots") or []
        if hotspots:
            top = hotspots[0]
            stitch_summary = top.get("stitch_summary") or {}
            candidates = top.get("stitched_candidates") or []
            if candidates:
                candidate = candidates[0]
                symbol = candidate.get("symbol") or {}
                symbol_desc = symbol.get("name") or candidate.get("entry_point")
                if symbol_desc and symbol.get("offset") is not None:
                    symbol_desc = f"{symbol_desc}+0x{int(symbol['offset']):x}"
                findings.append(
                    f"Top runtime hotspot maps back to code_object[{candidate['code_object_index']}] "
                    f"at {symbol_desc} via {candidate['match_kind']}."
                )
                share = stitch_summary.get("primary_dispatch_assignment_share")
                if isinstance(share, (int, float)):
                    findings.append(f"Primary hotspot candidate currently carries {share:.2f} of dispatch-span assignments in that stream.")
                dispatch_isa_share = stitch_summary.get("primary_dispatch_isa_mapped_dispatch_share")
                if isinstance(dispatch_isa_share, (int, float)):
                    findings.append(
                        f"Dispatch-segment ISA evidence maps {dispatch_isa_share:.2f} of sampled dispatch spans for that code object in this stream."
                    )
                top_pc = stitch_summary.get("primary_dispatch_isa_top_pc") or {}
                if top_pc:
                    findings.append(
                        "Dispatch-segment ISA evidence highlights "
                        f"pc=0x{int(top_pc.get('pc', 0)):x} {top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
                        + "."
                    )
                global_dispatch_isa_share = stitch_summary.get("primary_global_dispatch_isa_mapped_dispatch_share")
                if isinstance(global_dispatch_isa_share, (int, float)):
                    findings.append(
                        f"Capture-wide dispatch-segment ISA evidence maps {global_dispatch_isa_share:.2f} of sampled dispatch spans for that code object."
                    )
                global_top_pc = stitch_summary.get("primary_global_dispatch_isa_top_pc") or {}
                if global_top_pc and not top_pc:
                    findings.append(
                        "Capture-wide dispatch-segment ISA evidence highlights "
                        f"pc=0x{int(global_top_pc.get('pc', 0)):x} {global_top_pc.get('mnemonic')} {global_top_pc.get('operands')}".rstrip()
                        + "."
                    )
                if len(candidates) > 1:
                    findings.append(
                        f"Runtime hotspot still has {len(candidates)} candidate code objects in this stream-level view."
                    )
            elif top.get("code_object_id", 0) == 0:
                findings.append("SQTT hotspot stitching is incomplete: decoded hotspots still have code_object_id=0.")

    if decode_json:
        stitched_decode = decode_json.get("stitched") or {}
        decode_diagnostics = stitched_decode.get("decode_diagnostics") or {}
        if decode_diagnostics.get("likely_missing_codeobj_instrumentation"):
            findings.append(
                "Decoder diagnostics suggest this capture is missing the code-object instrumentation expected by rocprof trace decode."
            )
        info_counts = decode_diagnostics.get("info_counts") or {}
        if info_counts:
            findings.append(f"Decoder info records: {info_counts}.")
        dispatch_isa_overview = stitched_decode.get("dispatch_isa_overview") or {}
        if dispatch_isa_overview:
            mapped = int(dispatch_isa_overview.get("mapped_dispatch_count", 0) or 0)
            total = int(dispatch_isa_overview.get("dispatch_count", 0) or 0)
            findings.append(
                f"Dispatch-segment ISA mapping recovered concrete PCs for {mapped}/{total} sampled dispatch spans."
            )
            ordered = dispatch_isa_overview.get("ordered") or []
            if ordered:
                top = ordered[0]
                share = top.get("mapped_dispatch_share")
                share_text = f" ({share:.2f} of its dispatch spans)" if isinstance(share, (int, float)) else ""
                top_pc = top.get("top_pc") or {}
                pc_text = ""
                if top_pc:
                    pc_text = (
                        f" top_pc=0x{int(top_pc.get('pc', 0)):x} "
                        f"{top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
                    )
                findings.append(
                    f"Dispatch-segment ISA evidence is currently strongest for code_object[{int(top['code_object_index'])}]"
                    f"{share_text}.{pc_text}"
                )

    if stitch_model:
        resolved = int(stitch_model.get("resolved_entry_count", 0))
        partial = int(stitch_model.get("partially_resolved_entry_count", 0))
        dispatch_streams = int(stitch_model.get("dispatch_stream_count", 0))
        assigned_streams = int(stitch_model.get("assigned_stream_count", 0))
        dispatch_markers = int(stitch_model.get("dispatch_marker_count", 0))
        bind_markers = int(stitch_model.get("bind_marker_count", 0))
        distinct_bind_pipeline_hashes = int(stitch_model.get("distinct_bind_pipeline_hash_count", 0))
        api_marker_streams = int(stitch_model.get("api_marker_stream_count", 0))
        api_markers = int(stitch_model.get("api_marker_count", 0))
        dispatch_api_span_streams = int(stitch_model.get("dispatch_api_span_stream_count", 0))
        dispatch_api_spans = int(stitch_model.get("dispatch_api_span_count", 0))
        dispatch_span_assignments = int(stitch_model.get("dispatch_span_assignment_count", 0))
        dominant_dispatch_code_object_index = stitch_model.get("dominant_dispatch_code_object_index")
        dominant_dispatch_code_object_count = int(stitch_model.get("dominant_dispatch_code_object_count", 0))
        dominant_dispatch_code_object_share = stitch_model.get("dominant_dispatch_code_object_share")
        unmatched_api_begins = int(stitch_model.get("unmatched_api_begin_count", 0))
        command_buffer_streams = int(stitch_model.get("command_buffer_stream_count", 0))
        command_buffer_markers = int(stitch_model.get("command_buffer_marker_count", 0))
        command_buffer_span_streams = int(stitch_model.get("command_buffer_span_stream_count", 0))
        command_buffer_spans = int(stitch_model.get("command_buffer_span_count", 0))
        unmatched_command_buffer_begins = int(stitch_model.get("unmatched_command_buffer_begin_count", 0))
        cb_start_markers = int(stitch_model.get("cb_start_marker_count", 0))
        cb_end_markers = int(stitch_model.get("cb_end_marker_count", 0))
        barrier_markers = int(stitch_model.get("barrier_marker_count", 0))
        userdata_signature = stitch_model.get("tinygrad_userdata_signature")
        rocprof_instrumentation = stitch_model.get("rocprof_instrumentation") or {}
        if resolved:
            findings.append(f".rgp stitch model resolved {resolved} code-object entries through CO -> PSO -> Loader correlation.")
        if partial:
            findings.append(f".rgp stitch model still has {partial} partially resolved code-object entries.")
        if dispatch_streams:
            findings.append(
                f".rgp stitch model sees {dispatch_markers} dispatch markers across {dispatch_streams} SQTT streams."
            )
        if assigned_streams:
            findings.append(f".rgp stitch model assigned {assigned_streams} SQTT streams to a code object.")
        if bind_markers:
            findings.append(f".rgp stitch model also sees {bind_markers} bind-pipeline markers.")
        if distinct_bind_pipeline_hashes > max(1, resolved):
            findings.append(
                f".rgp stitch model sees {distinct_bind_pipeline_hashes} distinct bind-pipeline hashes, "
                f"which exceeds the {resolved} resolved code-object entries."
            )
        if api_marker_streams:
            findings.append(
                f".rgp stitch model recovers {api_markers} high-confidence API markers across {api_marker_streams} SQTT streams."
            )
            if dispatch_api_span_streams:
                findings.append(
                    f".rgp stitch model pairs {dispatch_api_spans} dispatch API spans across {dispatch_api_span_streams} SQTT streams."
                )
            if dispatch_span_assignments:
                findings.append(
                    f".rgp stitch model assigns {dispatch_span_assignments} dispatch spans to specific code objects."
                )
            if isinstance(dominant_dispatch_code_object_index, int) and dominant_dispatch_code_object_count > 0:
                share_text = (
                    f" ({dominant_dispatch_code_object_share:.2f} of assigned dispatch spans)"
                    if isinstance(dominant_dispatch_code_object_share, (int, float))
                    else ""
                )
                findings.append(
                    f"Dispatch-span ranking currently favors code_object[{dominant_dispatch_code_object_index}]"
                    f"{share_text}."
                )
            if unmatched_api_begins:
                findings.append(f".rgp stitch model still has {unmatched_api_begins} unmatched API begin markers.")
            streams = stitch_model.get("streams") or []
            first_lifecycle = next((stream.get("api_lifecycle") for stream in streams if stream.get("api_lifecycle")), None)
            if first_lifecycle:
                sequence = first_lifecycle.get("ordered_begin_sequence_unique") or []
                if sequence:
                    findings.append(
                        f"Recovered API lifecycle starts with {', '.join(sequence[:4])}."
                    )
        if command_buffer_streams:
            findings.append(
                f".rgp stitch model recovered {command_buffer_markers} command-buffer lifecycle markers across "
                f"{command_buffer_streams} SQTT streams."
            )
        if command_buffer_span_streams:
            findings.append(
                f".rgp stitch model pairs {command_buffer_spans} command-buffer spans across "
                f"{command_buffer_span_streams} SQTT streams."
            )
        if unmatched_command_buffer_begins:
            findings.append(
                f".rgp stitch model still has {unmatched_command_buffer_begins} unmatched command-buffer begin markers."
            )
        if cb_start_markers or cb_end_markers:
            findings.append(f".rgp stitch model sees CB_START={cb_start_markers} and CB_END={cb_end_markers}.")
        if barrier_markers:
            findings.append(f".rgp stitch model also recovers {barrier_markers} barrier markers.")
        if userdata_signature:
            findings.append(
                f"tinygrad packet decode infers userdata marker writes via hi_byte={userdata_signature['hi_byte']} subop={userdata_signature['subop']}."
            )
        if rocprof_instrumentation:
            if int(rocprof_instrumentation.get("total_enable_packets", 0)) == 0:
                findings.append("No rocprof code-object instrumentation enable packets were detected in the sampled SQTT streams.")
            else:
                findings.append(
                    f"rocprof code-object instrumentation enable packets were detected in {rocprof_instrumentation.get('streams_with_enable')} streams."
                )

    if marker_summary:
        totals = marker_summary.get("high_confidence") or []
        by_name_phase = {(item["api_name"], item["phase"]): item["count"] for item in totals}
        dispatch_begin = by_name_phase.get(("ApiCmdDispatch", "begin"), 0)
        push_constants = by_name_phase.get(("ApiCmdPushConstants", "begin"), 0) + by_name_phase.get(
            ("ApiCmdPushConstants", "end"), 0
        )
        write_timestamp = by_name_phase.get(("ApiCmdWriteTimestamp", "begin"), 0) + by_name_phase.get(
            ("ApiCmdWriteTimestamp", "end"), 0
        )
        if dispatch_begin:
            findings.append(f"Raw SQTT markers show {dispatch_begin} dispatch begin markers in the scanned streams.")
        if push_constants:
            findings.append(f"Raw SQTT markers include push-constant traffic: {push_constants} markers.")
        if write_timestamp:
            findings.append(f"Raw SQTT markers include timestamp commands: {write_timestamp} markers.")

    if marker_totals:
        totals = {item["marker_type"]: item["count"] for item in marker_totals.get("high_confidence", [])}
        event_markers = totals.get("EVENT", 0)
        barrier_start = totals.get("BARRIER_START", 0)
        barrier_end = totals.get("BARRIER_END", 0)
        pipeline_bind = totals.get("BIND_PIPELINE", 0)
        if event_markers:
            findings.append(f"Raw SQTT markers expose {event_markers} event markers with command IDs.")
        if barrier_start or barrier_end:
            findings.append(f"Raw SQTT markers expose barrier context: start={barrier_start} end={barrier_end}.")
        if pipeline_bind:
            findings.append(f"Raw SQTT markers expose {pipeline_bind} pipeline-bind markers.")

    if tinygrad_map:
        category_summary = tinygrad_map.get("category_summary") or []
        pc_summary = tinygrad_map.get("pc_summary") or []
        if category_summary:
            top_category = category_summary[0]
            findings.append(
                f"tinygrad static packet mapping is currently dominated by {top_category['category']} instructions "
                f"count={top_category['count']}."
            )
        if pc_summary:
            top = pc_summary[0]
            findings.append(
                f"tinygrad packet-to-ISA mapping currently hits pc=0x{int(top['pc']):x} "
                f"{top['mnemonic']} {top['operands']} count={top['count']}."
            )
            branch = next((item for item in pc_summary if item.get("category") == "branch"), None)
            memory = next((item for item in pc_summary if item.get("category") == "memory"), None)
            if branch:
                findings.append(
                    f"tinygrad mapping also surfaces an early branch at pc=0x{int(branch['pc']):x} "
                    f"{branch['mnemonic']} count={branch['count']}."
                )
            if memory:
                findings.append(
                    f"tinygrad mapping surfaces a memory instruction at pc=0x{int(memory['pc']):x} "
                    f"{memory['mnemonic']} count={memory['count']}."
                )

    if dispatch_isa_map:
        overall_pc_summary = dispatch_isa_map.get("overall_pc_summary") or []
        mapped_dispatches = [
            item for item in (dispatch_isa_map.get("dispatches") or []) if int(item.get("mapped_count", 0)) > 0
        ]
        if overall_pc_summary:
            top = overall_pc_summary[0]
            findings.append(
                f"Dispatch-segment ISA mapping highlights code_object[{top['code_object_index']}] "
                f"pc=0x{int(top['pc']):x} {top['mnemonic']} {top['operands']} count={top['count']}."
            )
        if mapped_dispatches:
            findings.append(
                f"Dispatch-segment ISA mapping produced packet-to-ISA matches for {len(mapped_dispatches)} dispatch spans on stream {dispatch_isa_map.get('stream_index')}."
            )

    if not findings:
        findings.append("No strong heuristic signal was detected from the available capture data.")
    return findings


def shader_triage(
    report: dict[str, Any],
    *,
    helper: Path | None = None,
    build_helper: bool = False,
    decoder_lib_dir: Path | None = None,
    isa_tool: str | None = None,
    readelf_tool: str | None = None,
    limit: int = 10,
    hotspot_limit: int = 8,
) -> dict[str, Any]:
    resolved_helper = resolve_helper_path(decode_repo_root(), helper, build_helper)
    signals = _triage_signals(
        report,
        helper=resolved_helper,
        decoder_lib_dir=decoder_lib_dir or default_decoder_lib_dir(),
        isa_tool=isa_tool,
        readelf_tool=readelf_tool,
        limit=limit,
        hotspot_limit=hotspot_limit,
    )
    decode_json = signals["runtime_signals"]["decode"]
    decode_stream = signals["runtime_signals"]["decode_stream"]
    stitch_model = signals["stitch_signals"]
    marker_summary = signals["marker_signals"]["general_api"]
    raw_marker_summary = signals["marker_signals"]["raw_markers"]
    resources = signals["static_signals"]["resource_summary"]
    isa = signals["static_signals"]["isa_summary"]
    tinygrad_map = signals["experimental_signals"]["tinygrad_isa_map"]
    dispatch_isa_map = signals["experimental_signals"]["dispatch_isa_map"]
    resource0 = resources[0] if resources else None
    isa0 = isa[0] if isa else None
    findings = _triage_findings(
        resource0,
        isa0,
        decode_json,
        decode_stream,
        marker_summary,
        raw_marker_summary,
        stitch_model,
        tinygrad_map,
        dispatch_isa_map,
    )

    return {
        **signals,
        "findings": findings,
    }
