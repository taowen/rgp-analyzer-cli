from __future__ import annotations

from typing import Any

from .api_context import (
    pair_barrier_spans,
    pair_command_buffer_spans,
    pair_stream_api_spans,
    summarize_stream_api_lifecycle,
    summarize_stream_api_phases,
    summarize_stream_command_context,
)
from .marker_scan import scan_general_api_markers, scan_markers
from .rgp_records import collect_pso_records
from .tinygrad_support.userdata_markers import reconstruct_tinygrad_userdata_markers


def _position_key(marker: dict[str, Any]) -> tuple[int, int]:
    packet_index = marker.get("packet_index")
    if isinstance(packet_index, int):
        return (0, packet_index)
    dword_index = marker.get("dword_index")
    if isinstance(dword_index, int):
        return (1, dword_index)
    byte_offset = marker.get("byte_offset")
    if isinstance(byte_offset, int):
        return (2, byte_offset)
    return (3, 0)


def _sorted_unique(markers: list[dict[str, Any]], identity_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    ordered: list[dict[str, Any]] = []
    for marker in sorted(markers, key=_position_key):
        key = tuple(marker.get(field) for field in identity_fields)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(marker)
    return ordered


def collect_stream_marker_context(report: dict[str, Any]) -> dict[str, Any]:
    marker_result = scan_markers(report, stream_limit=0)
    general_api_result = scan_general_api_markers(report, stream_limit=0)
    try:
        tinygrad_marker_result = reconstruct_tinygrad_userdata_markers(report, stream_limit=0)
    except Exception:
        tinygrad_marker_result = {"streams": [], "inferred_userdata_signature": None}

    known_api_pso_hashes = {
        int(record["api_pso_hash"])
        for record in collect_pso_records(report)
        if isinstance(record.get("api_pso_hash"), int)
    }
    general_api_by_stream = {stream["stream_index"]: stream for stream in general_api_result.get("streams", [])}
    tinygrad_by_stream = {stream["stream_index"]: stream for stream in tinygrad_marker_result.get("streams", [])}

    streams: list[dict[str, Any]] = []
    for stream in marker_result.get("streams", []):
        bind_markers = []
        dispatch_markers = []
        api_markers = []
        command_buffer_markers = []
        barrier_markers = []
        for marker in stream.get("markers", []):
            if marker.get("identifier") == 12 and marker.get("api_pso_hash") in known_api_pso_hashes:
                bind_markers.append(
                    {
                        "byte_offset": marker.get("byte_offset"),
                        "dword_index": marker.get("dword_index"),
                        "bind_point": marker.get("bind_point"),
                        "api_pso_hash": marker.get("api_pso_hash"),
                        "confidence": marker.get("confidence"),
                        "source": "raw_scan",
                    }
                )
            elif (
                marker.get("identifier") == 0
                and marker.get("confidence") == "high"
                and marker.get("event_name") in {"EventCmdDispatch", "EventCmdDispatchIndirect"}
            ):
                dispatch_markers.append(
                    {
                        "byte_offset": marker.get("byte_offset"),
                        "dword_index": marker.get("dword_index"),
                        "event_name": marker.get("event_name"),
                        "thread_dims": marker.get("thread_dims"),
                        "confidence": marker.get("confidence"),
                        "source": "raw_scan",
                    }
                )

        tinygrad_stream = tinygrad_by_stream.get(stream["stream_index"], {})
        prefer_tinygrad_api = bool(tinygrad_stream.get("markers"))
        general_api_stream = general_api_by_stream.get(stream["stream_index"], {})
        if not prefer_tinygrad_api:
            for marker in general_api_stream.get("markers", []):
                if marker.get("confidence") == "high":
                    api_markers.append(
                        {
                            "api_name": marker.get("api_name"),
                            "phase": marker.get("phase"),
                            "byte_offset": marker.get("byte_offset"),
                            "dword_index": marker.get("dword_index"),
                            "confidence": marker.get("confidence"),
                            "source": "general_api_scan",
                        }
                    )
                if marker.get("api_name") not in {"ApiCmdDispatch", "ApiCmdDispatchIndirect"} or marker.get("phase") != "begin":
                    continue
                dispatch_markers.append(
                    {
                        "byte_offset": marker.get("byte_offset"),
                        "dword_index": marker.get("dword_index"),
                        "event_name": marker.get("api_name"),
                        "thread_dims": None,
                        "confidence": marker.get("confidence"),
                        "source": "general_api_scan",
                    }
                )

        for marker in tinygrad_stream.get("markers", []):
            if marker.get("identifier") == 12 and marker.get("api_pso_hash") in known_api_pso_hashes:
                bind_markers.append(
                    {
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "bind_point": marker.get("bind_point"),
                        "api_pso_hash": marker.get("api_pso_hash"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
            elif marker.get("identifier") == 6 and marker.get("confidence") == "high":
                api_markers.append(
                    {
                        "api_name": marker.get("api_name"),
                        "phase": marker.get("phase"),
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
                if marker.get("api_name") not in {"ApiCmdDispatch", "ApiCmdDispatchIndirect"} or marker.get("phase") != "begin":
                    continue
                dispatch_markers.append(
                    {
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "dword_index": marker.get("dword_index"),
                        "event_name": marker.get("api_name"),
                        "thread_dims": marker.get("thread_dims"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
            elif (
                marker.get("identifier") == 0
                and marker.get("confidence") == "high"
                and marker.get("event_name") in {"EventCmdDispatch", "EventCmdDispatchIndirect"}
            ):
                dispatch_markers.append(
                    {
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "dword_index": marker.get("dword_index"),
                        "event_name": marker.get("event_name"),
                        "thread_dims": marker.get("thread_dims"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
            elif marker.get("identifier") == 1:
                if marker.get("confidence") != "high":
                    continue
                command_buffer_markers.append(
                    {
                        "kind": "CB_START",
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "cb_id": marker.get("cb_id"),
                        "device_id": marker.get("device_id"),
                        "queue": marker.get("queue"),
                        "queue_flags": marker.get("queue_flags"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
            elif marker.get("identifier") == 2:
                if marker.get("confidence") != "high":
                    continue
                command_buffer_markers.append(
                    {
                        "kind": "CB_END",
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "cb_id": marker.get("cb_id"),
                        "device_id": marker.get("device_id"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )
            elif marker.get("identifier") in {3, 4}:
                if marker.get("confidence") not in {"medium", "high"}:
                    continue
                barrier_markers.append(
                    {
                        "kind": marker.get("identifier_name"),
                        "packet_index": marker.get("packet_index"),
                        "byte_offset": marker.get("byte_offset"),
                        "cb_id": marker.get("cb_id"),
                        "driver_reason": marker.get("driver_reason"),
                        "internal": marker.get("internal"),
                        "num_layout_transitions": marker.get("num_layout_transitions"),
                        "confidence": marker.get("confidence"),
                        "source": marker.get("source"),
                    }
                )

        bind_markers = _sorted_unique(
            bind_markers,
            ("source", "api_pso_hash", "bind_point", "packet_index", "dword_index", "byte_offset"),
        )
        dispatch_markers = _sorted_unique(
            dispatch_markers,
            ("source", "event_name", "packet_index", "dword_index", "byte_offset"),
        )
        api_markers = _sorted_unique(
            api_markers,
            ("source", "api_name", "phase", "packet_index", "dword_index", "byte_offset"),
        )
        command_buffer_markers = _sorted_unique(
            command_buffer_markers,
            ("source", "kind", "cb_id", "device_id", "packet_index", "dword_index", "byte_offset"),
        )
        barrier_markers = _sorted_unique(
            barrier_markers,
            ("source", "kind", "cb_id", "packet_index", "dword_index", "byte_offset"),
        )
        command_context = summarize_stream_command_context(command_buffer_markers, barrier_markers)
        command_spans = pair_command_buffer_spans(command_buffer_markers)
        barrier_spans = pair_barrier_spans(barrier_markers)
        api_summary = summarize_stream_api_phases(api_markers)
        api_lifecycle = summarize_stream_api_lifecycle(api_markers)
        api_spans = pair_stream_api_spans(api_markers)

        streams.append(
            {
                "stream_index": stream["stream_index"],
                "shader_engine_index": stream.get("shader_engine_index"),
                "compute_unit_index": stream.get("compute_unit_index"),
                "bind_pipeline_markers": bind_markers,
                "dispatch_event_markers": dispatch_markers,
                "api_markers": api_markers,
                "command_buffer_markers": command_buffer_markers,
                "barrier_markers": barrier_markers,
                "command_buffer_ids": command_context["command_buffer_ids"],
                "command_buffer_spans": command_spans["command_buffer_spans"],
                "unmatched_command_buffer_begins": command_spans["unmatched_command_buffer_begins"],
                "barrier_spans": barrier_spans["barrier_spans"],
                "unmatched_barrier_begins": barrier_spans["unmatched_barrier_begins"],
                "api_marker_summary": api_summary,
                "api_lifecycle": api_lifecycle,
                "api_spans": api_spans["api_spans"],
                "dispatch_api_spans": api_spans["dispatch_spans"],
                "unmatched_api_begin_markers": api_spans["unmatched_begin_markers"],
                "command_buffer_summary": command_context["command_buffer_summary"],
            }
        )

    return {
        "streams": streams,
        "tinygrad_userdata_signature": tinygrad_marker_result.get("inferred_userdata_signature"),
        "rocprof_instrumentation": tinygrad_marker_result.get("rocprof_instrumentation"),
    }
