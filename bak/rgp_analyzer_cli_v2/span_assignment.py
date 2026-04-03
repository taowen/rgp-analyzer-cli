from __future__ import annotations

from typing import Any


def _position(marker: dict[str, Any] | None) -> tuple[int, int]:
    if not marker:
        return (9, 0)
    packet_index = marker.get("packet_index")
    if isinstance(packet_index, int):
        return (0, packet_index)
    dword_index = marker.get("dword_index")
    if isinstance(dword_index, int):
        return (1, dword_index)
    byte_offset = marker.get("byte_offset")
    if isinstance(byte_offset, int):
        return (2, byte_offset)
    return (9, 0)


def assign_dispatch_spans_to_entries(
    entries: list[dict[str, Any]],
    stream_markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entry_by_api_hash = {entry.get("api_pso_hash"): entry for entry in entries if entry.get("resolved")}

    for stream in stream_markers:
        bind_markers = sorted(stream.get("bind_pipeline_markers", []), key=_position)
        dispatch_assignments: list[dict[str, Any]] = []
        for span in stream.get("dispatch_api_spans", []):
            begin_marker = span.get("begin_marker")
            begin_pos = _position(begin_marker)
            candidate = None
            for bind in bind_markers:
                if _position(bind) <= begin_pos:
                    candidate = bind
                else:
                    break
            entry = entry_by_api_hash.get(candidate.get("api_pso_hash")) if candidate else None
            dispatch_assignments.append(
                {
                    "api_name": span.get("api_name"),
                    "matched": bool(entry),
                    "bind_api_pso_hash": candidate.get("api_pso_hash") if candidate else None,
                    "bind_packet_index": candidate.get("packet_index") if candidate else None,
                    "bind_byte_offset": candidate.get("byte_offset") if candidate else None,
                    "code_object_index": entry.get("code_object_index") if entry else None,
                    "entry_point": entry.get("entry_point") if entry else None,
                    "match_kind": "dispatch_span_preceding_bind" if entry else None,
                    "begin_position": begin_pos,
                    "end_position": _position(span.get("end_marker")),
                }
            )
        stream["dispatch_assignments"] = dispatch_assignments
        stream["assigned_dispatch_span_count"] = sum(1 for item in dispatch_assignments if item.get("matched"))
        histogram: dict[int, int] = {}
        for item in dispatch_assignments:
            code_object_index = item.get("code_object_index")
            if isinstance(code_object_index, int):
                histogram[code_object_index] = histogram.get(code_object_index, 0) + 1
        stream["dispatch_assignment_histogram"] = histogram
        stream["distinct_assigned_code_objects"] = sorted(
            {
                int(item["code_object_index"])
                for item in dispatch_assignments
                if isinstance(item.get("code_object_index"), int)
            }
        )

    return stream_markers
