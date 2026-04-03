from __future__ import annotations

from typing import Any

from .marker_sources import collect_stream_marker_context
from .span_assignment import assign_dispatch_spans_to_entries
from .stitch_records import build_stitch_entries
from .stream_assignment import assign_streams_to_entries, mark_entries_seen_in_bind_markers


def _stream_stats(stream_markers: list[dict[str, Any]]) -> dict[str, int]:
    distinct_bind_hashes = {
        int(marker["api_pso_hash"])
        for stream in stream_markers
        for marker in stream.get("bind_pipeline_markers", [])
        if isinstance(marker.get("api_pso_hash"), int)
    }
    return {
        "dispatch_stream_count": sum(1 for stream in stream_markers if stream.get("dispatch_event_markers")),
        "assigned_stream_count": sum(1 for stream in stream_markers if stream.get("resolved_code_object_index") is not None),
        "dispatch_marker_count": sum(len(stream.get("dispatch_event_markers", [])) for stream in stream_markers),
        "bind_marker_count": sum(len(stream.get("bind_pipeline_markers", [])) for stream in stream_markers),
        "api_marker_stream_count": sum(1 for stream in stream_markers if stream.get("api_markers")),
        "api_marker_count": sum(len(stream.get("api_markers", [])) for stream in stream_markers),
        "dispatch_api_span_stream_count": sum(1 for stream in stream_markers if stream.get("dispatch_api_spans")),
        "dispatch_api_span_count": sum(len(stream.get("dispatch_api_spans", [])) for stream in stream_markers),
        "unmatched_api_begin_count": sum(len(stream.get("unmatched_api_begin_markers", [])) for stream in stream_markers),
        "command_buffer_stream_count": sum(1 for stream in stream_markers if stream.get("command_buffer_markers")),
        "command_buffer_marker_count": sum(len(stream.get("command_buffer_markers", [])) for stream in stream_markers),
        "command_buffer_span_stream_count": sum(1 for stream in stream_markers if stream.get("command_buffer_spans")),
        "command_buffer_span_count": sum(len(stream.get("command_buffer_spans", [])) for stream in stream_markers),
        "unmatched_command_buffer_begin_count": sum(
            len(stream.get("unmatched_command_buffer_begins", [])) for stream in stream_markers
        ),
        "cb_start_marker_count": sum(
            1
            for stream in stream_markers
            for marker in stream.get("command_buffer_markers", [])
            if marker.get("kind") == "CB_START"
        ),
        "cb_end_marker_count": sum(
            1
            for stream in stream_markers
            for marker in stream.get("command_buffer_markers", [])
            if marker.get("kind") == "CB_END"
        ),
        "barrier_marker_count": sum(len(stream.get("barrier_markers", [])) for stream in stream_markers),
        "barrier_start_marker_count": sum(
            1
            for stream in stream_markers
            for marker in stream.get("barrier_markers", [])
            if marker.get("kind") == "BARRIER_START"
        ),
        "barrier_end_marker_count": sum(
            1
            for stream in stream_markers
            for marker in stream.get("barrier_markers", [])
            if marker.get("kind") == "BARRIER_END"
        ),
        "distinct_bind_pipeline_hash_count": len(distinct_bind_hashes),
        "dispatch_span_assignment_count": sum(
            int(stream.get("assigned_dispatch_span_count", 0)) for stream in stream_markers
        ),
    }


def _dispatch_assignment_summary(stream_markers: list[dict[str, Any]]) -> dict[str, Any]:
    histogram: dict[int, int] = {}
    for stream in stream_markers:
        for code_object_index, count in (stream.get("dispatch_assignment_histogram") or {}).items():
            if isinstance(code_object_index, int) and isinstance(count, int):
                histogram[code_object_index] = histogram.get(code_object_index, 0) + count

    ordered = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    total = sum(histogram.values())
    dominant_index = ordered[0][0] if ordered else None
    dominant_count = ordered[0][1] if ordered else 0
    dominant_share = (dominant_count / total) if total else None
    return {
        "dispatch_assignment_histogram": histogram,
        "dispatch_assignment_histogram_ordered": [
            {"code_object_index": code_object_index, "count": count, "share": (count / total) if total else None}
            for code_object_index, count in ordered
        ],
        "dominant_dispatch_code_object_index": dominant_index,
        "dominant_dispatch_code_object_count": dominant_count,
        "dominant_dispatch_code_object_share": dominant_share,
    }


def build_stitch_model(report: dict[str, Any]) -> dict[str, Any]:
    records = build_stitch_entries(report)
    entries = records["entries"]
    marker_context = collect_stream_marker_context(report)
    stream_markers = marker_context["streams"]

    mark_entries_seen_in_bind_markers(entries, stream_markers)
    assign_streams_to_entries(entries, stream_markers)
    assign_dispatch_spans_to_entries(entries, stream_markers)
    stats = _stream_stats(stream_markers)
    assignment_summary = _dispatch_assignment_summary(stream_markers)
    rocprof_instrumentation = marker_context.get("rocprof_instrumentation") or {}
    if stats["distinct_bind_pipeline_hash_count"] > max(1, records["resolved_entry_count"]):
        records["warnings"].append(
            "Capture contains more distinct bind-pipeline hashes than resolved code-object entries; stitching may still be under-resolved."
        )
    if int(rocprof_instrumentation.get("total_enable_packets", 0)) == 0:
        records["warnings"].append(
            "No rocprof code-object instrumentation enable packets were detected in the sampled SQTT streams."
        )

    return {
        "model_kind": "co_col_pso",
        "code_object_record_count": len(records["code_records"]),
        "loader_load_record_count": len(records["loader_records"]),
        "pso_record_count": len(records["pso_records"]),
        "metadata_record_count": len(records["resource_metadata"]),
        "resolved_entry_count": records["resolved_entry_count"],
        "partially_resolved_entry_count": len(entries) - records["resolved_entry_count"],
        **stats,
        **assignment_summary,
        "tinygrad_userdata_signature": marker_context.get("tinygrad_userdata_signature"),
        "rocprof_instrumentation": rocprof_instrumentation,
        "warnings": records["warnings"],
        "entries": entries,
        "streams": stream_markers,
    }
