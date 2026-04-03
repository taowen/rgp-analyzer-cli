from __future__ import annotations

from typing import Any

from .analyze import (
    chunk_counts,
    code_object_summary,
    compare_reports,
    decoder_bridge,
    correlation_summary,
    flattened_events,
    generate_advice,
    pso_summary,
    scan_sqtt_payload,
    sqtt_summary,
    summarize_isa_text,
)
from .decode_bridge import build_stitch_model


def render_inspect(report: dict[str, Any], limit: int = 10) -> str:
    header = report["header"]
    ts = header["timestamp"]
    lines = [
        f"file: {report['file']}",
        f"size_bytes: {report['size_bytes']}",
        f"magic: {header['magic_hex']} valid={header['valid_magic']}",
        f"version: {header['version_major']}.{header['version_minor']}",
        f"captured_at: {ts['year']:04d}-{ts['month']:02d}-{ts['day']:02d} {ts['hour']:02d}:{ts['minute']:02d}:{ts['second']:02d}",
        "",
        "chunk_counts:",
    ]
    for name, count in chunk_counts(report).items():
        lines.append(f"  {name}: {count}")

    events = sorted(flattened_events(report), key=lambda item: item["gpu_duration"], reverse=True)
    lines.extend(["", f"top_events_by_gpu_duration: {min(limit, len(events))}"])
    for event in events[:limit]:
        lines.append(
            "  "
            + f"{event['event_name']:<17} gpu_duration={event['gpu_duration']:<12} "
            + f"queue={event['queue_type_name']}/{event['engine_type_name']:<18} "
            + f"submit={event['submit_sub_index']:<4} frame={event['frame_index']}"
        )

    code_objects = code_object_summary(report)
    lines.extend(
        [
            "",
            "code_objects:",
            f"  databases: {code_objects['database_count']}",
            f"  records: {code_objects['record_count']}",
            f"  elf_records: {code_objects['elf_record_count']}",
            f"  total_payload_bytes: {code_objects['total_payload_bytes']}",
        ]
    )

    pso = pso_summary(report)
    sqtt = sqtt_summary(report)
    lines.extend(
        [
            "",
            "sqtt:",
            f"  descs: {sqtt['desc_count']}",
            f"  data_chunks: {sqtt['data_chunk_count']}",
            f"  total_trace_bytes: {sqtt['total_trace_bytes']}",
            "",
            "pso_correlation:",
            f"  chunks: {pso['chunk_count']}",
            f"  records: {pso['record_count']}",
            f"  named_records: {len(pso['named_records'])}",
        ]
    )

    lines.extend(["", "notes:"])
    for item in generate_advice(report):
        lines.append(f"  - {item}")

    return "\n".join(lines)


def render_chunks(report: dict[str, Any]) -> str:
    lines = ["chunks:"]
    for chunk in report["chunks"]:
        lines.append(
            f"  off={chunk['offset']:>10} size={chunk['size_in_bytes']:>10} "
            f"type={chunk['type_name']}#{chunk['index']} v{chunk['major_version']}.{chunk['minor_version']}"
        )
    return "\n".join(lines)


def render_events(report: dict[str, Any], limit: int, sort_key: str) -> str:
    events = flattened_events(report)
    events = sorted(events, key=lambda item: item.get(sort_key, 0), reverse=True)
    lines = ["events:"]
    for event in events[:limit]:
        lines.append(
            "  "
            + f"{event['event_name']:<17} gpu_duration={event['gpu_duration']:<12} "
            + f"queue={event['queue_type_name']}/{event['engine_type_name']:<18} "
            + f"submit={event['submit_sub_index']:<4} frame={event['frame_index']:<4} "
            + f"gpu0={event['gpu_timestamp_0']} gpu1={event['gpu_timestamp_1']}"
        )
    return "\n".join(lines)


def render_code_objects(report: dict[str, Any], show_strings: bool, limit: int) -> str:
    records = code_object_summary(report)["largest_records"][:limit]
    lines = ["code_objects:"]
    for record in records:
        lines.append(
            f"  index={record['index']:<4} payload_size={record['payload_size']:<10} record_size={record['record_size']}"
        )
        elf = record.get("elf", {})
        if elf.get("valid_elf"):
            lines.append(
                f"    elf sections={elf['section_count']} symbols={len(elf['symbols'])} notes={len(elf['notes'])} flags=0x{elf['flags']:x}"
            )
            section_names = [section["name"] for section in elf["sections"][:12] if section["name"]]
            if section_names:
                lines.append(f"    sections={', '.join(section_names)}")
            symbol_names = [symbol["name"] for symbol in elf["symbols"][:10]]
            if symbol_names:
                lines.append(f"    symbols={', '.join(symbol_names)}")
            note_names = [note["name"] for note in elf["notes"][:10] if note["name"]]
            if note_names:
                lines.append(f"    notes={', '.join(note_names)}")
        if show_strings:
            for value in record["embedded_strings"][:10]:
                lines.append(f"    string={value}")
    return "\n".join(lines)


def render_pso(report: dict[str, Any], limit: int) -> str:
    records = []
    for chunk in report["pso_correlations"]:
        records.extend(chunk["records"])
    lines = ["pso_correlation:"]
    for record in records[:limit]:
        lines.append(
            f"  api_pso_hash={record['api_pso_hash']} pipeline_hash={record['pipeline_hash'][0]:016x}:{record['pipeline_hash'][1]:016x} "
            + f"name={record['api_level_obj_name']}"
        )
    return "\n".join(lines)


def render_loader_events(report: dict[str, Any], limit: int) -> str:
    records = []
    for chunk in report["loader_events"]:
        records.extend(chunk["records"])
    lines = ["loader_events:"]
    for record in records[:limit]:
        lines.append(
            f"  {record['loader_event_type_name']:<20} base_address=0x{record['base_address']:x} "
            + f"time_stamp={record['time_stamp']} hash={record['code_object_hash'][0]:016x}:{record['code_object_hash'][1]:016x}"
        )
    return "\n".join(lines)


def render_advice(report: dict[str, Any]) -> str:
    lines = ["notes:"]
    for item in generate_advice(report):
        lines.append(f"  - {item}")
    return "\n".join(lines)


def render_sqtt(report: dict[str, Any], limit: int) -> str:
    lines = ["sqtt:"]
    for index, desc in enumerate(report["sqtt_descs"][:limit]):
        lines.append(
            f"  desc[{index}] shader_engine={desc['shader_engine_index']} cu={desc['compute_unit_index']} "
            + f"sqtt_version={desc['sqtt_version']} instr_spec={desc['instrumentation_spec_version']} api={desc['instrumentation_api_version']}"
        )
    for index, data in enumerate(report["sqtt_data_chunks"][:limit]):
        lines.append(
            f"  data[{index}] offset={data['offset']} size={data['size']} payload_offset={data['payload_offset']} payload_end={data['payload_end']}"
        )
    return "\n".join(lines)


def render_correlate(report: dict[str, Any], limit: int) -> str:
    corr = correlation_summary(report)
    lines = [
        "correlation:",
        f"  loader_records: {corr['loader_record_count']}",
        f"  pso_records: {corr['pso_record_count']}",
        f"  matched_pso: {corr['matched_pso_count']}",
        f"  unmatched_pso: {corr['unmatched_pso_count']}",
    ]
    for item in corr["correlations"][:limit]:
        lines.append(
            "  "
            + f"pipeline_hash={item['pipeline_hash'][0]:016x}:{item['pipeline_hash'][1]:016x} "
            + f"loader_match_count={item['loader_match_count']} "
            + f"bases={[hex(v) for v in item['loader_base_addresses']]} "
            + f"name={item['api_level_obj_name']}"
        )
    return "\n".join(lines)


def render_compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    diff = compare_reports(baseline, candidate)

    def fmt_delta(entry: dict[str, Any]) -> str:
        percent = entry["delta"]["percent"]
        pct = "n/a" if percent is None else f"{percent:.2f}%"
        return f"baseline={entry['baseline']} candidate={entry['candidate']} delta={entry['delta']['absolute']} ({pct})"

    lines = [
        "compare:",
        f"  baseline: {diff['baseline_file']}",
        f"  candidate: {diff['candidate_file']}",
        f"  total_gpu_duration: {fmt_delta(diff['total_gpu_duration'])}",
        f"  max_gpu_duration: {fmt_delta(diff['max_gpu_duration'])}",
        f"  avg_gpu_duration: {fmt_delta(diff['avg_gpu_duration'])}",
        f"  code_object_records: baseline={diff['code_object_records']['baseline']} candidate={diff['code_object_records']['candidate']} delta={diff['code_object_records']['delta']}",
        f"  sqtt_trace_bytes: {fmt_delta(diff['sqtt_trace_bytes'])}",
        f"  queue_types: baseline={diff['queue_types']['baseline']} candidate={diff['queue_types']['candidate']}",
        "  chunk_counts_baseline:",
    ]
    for name, count in diff["chunk_counts"]["baseline"].items():
        lines.append(f"    {name}: {count}")
    lines.append("  chunk_counts_candidate:")
    for name, count in diff["chunk_counts"]["candidate"].items():
        lines.append(f"    {name}: {count}")
    return "\n".join(lines)


def render_scan_sqtt(report: dict[str, Any], stream_limit: int, dword_limit: int) -> str:
    lines = ["scan_sqtt:"]
    for index in range(min(stream_limit, len(report["sqtt_data_chunks"]))):
        scan = scan_sqtt_payload(report, index=index, dword_limit=dword_limit)
        lines.append(
            f"  stream[{index}] size_bytes={scan['size_bytes']} dword_count={scan['dword_count']} "
            + f"high_nibbles={scan['high_nibble_histogram']}"
        )
        preview = " ".join(f"{value:08x}" for value in scan["preview_dwords"])
        lines.append(f"    preview_dwords={preview}")
    return "\n".join(lines)


def render_isa_summary(results: list[dict[str, Any]]) -> str:
    lines = ["isa_summary:"]
    for item in results:
        summary = summarize_isa_text(item["stdout"])
        lines.append(
            f"  code_object[{item['index']}] instructions={summary['instruction_count']} unique_opcodes={summary['unique_opcodes']}"
        )
        lines.append(
            f"    vector_alu={summary['vector_alu']} scalar_ops={summary['scalar_ops']} "
            + f"buffer_ops={summary['buffer_ops']} lds_ops={summary['lds_ops']} "
            + f"flat_ops={summary['flat_ops']} image_ops={summary['image_ops']} "
            + f"wait_ops={summary['wait_ops']} branch_ops={summary['branch_ops']}"
        )
        top = ", ".join(f"{name}:{count}" for name, count in list(summary["opcode_counts"].items())[:12])
        lines.append(f"    top_opcodes={top}")
    return "\n".join(lines)


def render_decoder_bridge(report: dict[str, Any], limit: int = 10) -> str:
    bridge = decoder_bridge(report)
    lines = [
        "decoder_bridge:",
        f"  code_object_records: {bridge['code_object_record_count']}",
        f"  loader_load_records: {bridge['loader_load_record_count']}",
        f"  pso_records: {bridge['pso_record_count']}",
        f"  bridged_entries: {bridge['bridged_count']}",
        f"  resolved_entries: {bridge['resolved_entry_count']}",
    ]
    for warning in bridge["warnings"]:
        lines.append(f"  warning: {warning}")
    for item in bridge["entries"][:limit]:
        lines.append(
            "  "
            + f"bridge[{item['bridge_index']}] code_object={item['code_object_index']} "
            + f"load_id={item['load_id']} load_addr=0x{item['load_addr']:x} "
            + f"load_size={item['load_size']} matched_loader={item['matched_loader_event']} "
            + f"api_pso_hash={item.get('api_pso_hash')} seen_in_bind={item.get('seen_in_bind_pipeline_markers')}"
        )
    return "\n".join(lines)


def render_stitch_report(report: dict[str, Any], limit: int = 10) -> str:
    model = build_stitch_model(report)
    lines = [
        "stitch_report:",
        f"  model_kind: {model['model_kind']}",
        f"  code_object_records: {model['code_object_record_count']}",
        f"  pso_records: {model['pso_record_count']}",
        f"  loader_load_records: {model['loader_load_record_count']}",
        f"  metadata_records: {model['metadata_record_count']}",
        f"  resolved_entries: {model['resolved_entry_count']}",
        f"  partially_resolved_entries: {model['partially_resolved_entry_count']}",
        f"  dispatch_streams: {model['dispatch_stream_count']}",
        f"  assigned_streams: {model['assigned_stream_count']}",
        f"  dispatch_markers: {model['dispatch_marker_count']}",
        f"  bind_markers: {model['bind_marker_count']}",
        f"  distinct_bind_pipeline_hashes: {model['distinct_bind_pipeline_hash_count']}",
        f"  api_marker_streams: {model['api_marker_stream_count']}",
        f"  api_markers: {model['api_marker_count']}",
        f"  dispatch_api_span_streams: {model['dispatch_api_span_stream_count']}",
        f"  dispatch_api_spans: {model['dispatch_api_span_count']}",
        f"  dispatch_span_assignments: {model['dispatch_span_assignment_count']}",
        f"  dispatch_assignment_histogram: {model.get('dispatch_assignment_histogram')}",
        f"  dominant_dispatch_code_object: {model.get('dominant_dispatch_code_object_index')}",
        f"  dominant_dispatch_code_object_count: {model.get('dominant_dispatch_code_object_count')}",
        f"  dominant_dispatch_code_object_share: {model.get('dominant_dispatch_code_object_share')}",
        f"  unmatched_api_begins: {model['unmatched_api_begin_count']}",
        f"  command_buffer_streams: {model['command_buffer_stream_count']}",
        f"  command_buffer_markers: {model['command_buffer_marker_count']}",
        f"  command_buffer_span_streams: {model['command_buffer_span_stream_count']}",
        f"  command_buffer_spans: {model['command_buffer_span_count']}",
        f"  unmatched_command_buffer_begins: {model['unmatched_command_buffer_begin_count']}",
        f"  cb_start_markers: {model['cb_start_marker_count']}",
        f"  cb_end_markers: {model['cb_end_marker_count']}",
        f"  barrier_markers: {model['barrier_marker_count']}",
        f"  tinygrad_userdata_signature: {model.get('tinygrad_userdata_signature')}",
        f"  rocprof_instrumentation: {model.get('rocprof_instrumentation')}",
    ]
    for warning in model["warnings"]:
        lines.append(f"  warning: {warning}")
    if model.get("dispatch_assignment_histogram_ordered"):
        lines.append("  dispatch_assignment_ranking:")
        for item in model["dispatch_assignment_histogram_ordered"][:limit]:
            share = item.get("share")
            share_text = f" share={share:.2f}" if share is not None else ""
            lines.append(
                "    "
                + f"code_object[{item['code_object_index']}] count={item['count']}{share_text}"
            )
    lines.append("  entries:")
    for entry in model["entries"][:limit]:
        lines.append(
            "    "
            + f"code_object[{entry['code_object_index']}] entry_point={entry.get('entry_point')} "
            + f"internal_pipeline_hash={entry.get('internal_pipeline_hash_text')} "
            + f"api_pso_hash={entry.get('api_pso_hash')} "
            + f"pipeline_hash={entry.get('pipeline_hash_text')} "
            + f"loader_matches={entry.get('loader_match_count')} "
            + f"resolved={entry.get('resolved')} seen_in_bind={entry.get('seen_in_bind_pipeline_markers')}"
        )
        if entry.get("resolution_steps"):
            lines.append(f"      steps={' ; '.join(entry['resolution_steps'])}")
        if entry.get("unresolved_reasons"):
            lines.append(f"      unresolved={','.join(entry['unresolved_reasons'])}")
        for loader in entry.get("loader_records", [])[:2]:
            lines.append(
                "      "
                + f"loader[{loader['index']}] base=0x{loader['base_address']:x} "
                + f"time_stamp={loader['time_stamp']}"
            )
    lines.append("  streams:")
    for stream in model["streams"][:limit]:
        lines.append(
            "    "
            + f"stream[{stream['stream_index']}] se={stream.get('shader_engine_index')} "
                + f"cu={stream.get('compute_unit_index')} "
                + f"bind_markers={len(stream.get('bind_pipeline_markers', []))} "
                + f"dispatch_markers={len(stream.get('dispatch_event_markers', []))} "
                + f"api_markers={len(stream.get('api_markers', []))} "
                + f"dispatch_spans={len(stream.get('dispatch_api_spans', []))} "
                + f"dispatch_assigned={stream.get('assigned_dispatch_span_count', 0)} "
                + f"cb_markers={len(stream.get('command_buffer_markers', []))} "
                + f"cb_spans={len(stream.get('command_buffer_spans', []))} "
                + f"barrier_markers={len(stream.get('barrier_markers', []))} "
                + f"resolved_code_object={stream.get('resolved_code_object_index')} "
                + f"stream_match={stream.get('stream_match_kind')}"
        )
        for marker in stream.get("bind_pipeline_markers", [])[:2]:
            lines.append(
                "      "
                + f"bind api_pso_hash={marker.get('api_pso_hash')} bind_point={marker.get('bind_point')} "
                + f"byte_offset={marker.get('byte_offset')}"
            )
        for marker in stream.get("dispatch_event_markers", [])[:2]:
            dims = marker.get("thread_dims") or {}
            lines.append(
                "      "
                + f"{marker.get('event_name')} dims={dims.get('x')}x{dims.get('y')}x{dims.get('z')} "
                + f"byte_offset={marker.get('byte_offset')}"
            )
        for marker in stream.get("api_markers", [])[:4]:
            lines.append(
                "      "
                + f"{marker.get('api_name')} phase={marker.get('phase')} "
                + f"byte_offset={marker.get('byte_offset')} source={marker.get('source')}"
            )
        lifecycle = stream.get("api_lifecycle") or {}
        sequence = lifecycle.get("ordered_begin_sequence_unique") or []
        if sequence:
            lines.append("      " + f"api_sequence={', '.join(sequence[:6])}")
        spans = stream.get("dispatch_api_spans") or []
        if spans:
            lines.append("      " + f"dispatch_api_spans={len(spans)}")
        assignments = stream.get("dispatch_assignments") or []
        if assignments:
            lines.append(
                "      "
                + f"dispatch_assigned_code_objects={stream.get('distinct_assigned_code_objects')}"
            )
        cb_spans = stream.get("command_buffer_spans") or []
        if cb_spans:
            lines.append("      " + f"command_buffer_spans={len(cb_spans)}")
        for marker in stream.get("command_buffer_markers", [])[:2]:
            lines.append(
                "      "
                + f"{marker.get('kind')} cb_id={marker.get('cb_id')} queue={marker.get('queue')} "
                + f"byte_offset={marker.get('byte_offset')} source={marker.get('source')}"
            )
        for marker in stream.get("barrier_markers", [])[:2]:
            lines.append(
                "      "
                + f"{marker.get('kind')} cb_id={marker.get('cb_id')} driver_reason={marker.get('driver_reason')} "
                + f"byte_offset={marker.get('byte_offset')} source={marker.get('source')}"
            )
    return "\n".join(lines)
