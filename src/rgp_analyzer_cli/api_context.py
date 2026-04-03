from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_stream_api_phases(api_markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter((marker["api_name"], marker["phase"]) for marker in api_markers)
    return [
        {"api_name": api_name, "phase": phase, "count": count}
        for (api_name, phase), count in sorted(totals.items())
    ]


def summarize_stream_api_lifecycle(api_markers: list[dict[str, Any]]) -> dict[str, Any]:
    names = [marker["api_name"] for marker in api_markers if marker.get("phase") == "begin" and marker.get("api_name")]
    begin_counts = Counter(marker["api_name"] for marker in api_markers if marker.get("phase") == "begin" and marker.get("api_name"))
    return {
        "ordered_begin_sequence": names,
        "ordered_begin_sequence_unique": list(dict.fromkeys(names)),
        "begin_counts": [{"api_name": api_name, "count": count} for api_name, count in sorted(begin_counts.items())],
    }


def _marker_position(marker: dict[str, Any]) -> tuple[int, int]:
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


def pair_stream_api_spans(api_markers: list[dict[str, Any]]) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    open_spans: dict[str, list[dict[str, Any]]] = {}
    for marker in sorted(api_markers, key=_marker_position):
        api_name = marker.get("api_name")
        phase = marker.get("phase")
        if not api_name or phase not in {"begin", "end"}:
            continue
        if phase == "begin":
            open_spans.setdefault(api_name, []).append(marker)
            continue
        begin_list = open_spans.get(api_name)
        begin_marker = begin_list.pop() if begin_list else None
        span = {
            "api_name": api_name,
            "matched": begin_marker is not None,
            "begin_marker": begin_marker,
            "end_marker": marker,
        }
        if begin_marker is not None:
            span["begin_position"] = _marker_position(begin_marker)
            span["end_position"] = _marker_position(marker)
        spans.append(span)

    unmatched_begin = [
        {"api_name": api_name, "marker": marker}
        for api_name, markers in open_spans.items()
        for marker in markers
    ]
    dispatch_spans = [span for span in spans if span["api_name"] in {"ApiCmdDispatch", "ApiCmdDispatchIndirect"}]
    return {
        "api_spans": spans,
        "dispatch_spans": dispatch_spans,
        "unmatched_begin_markers": unmatched_begin,
    }


def summarize_stream_command_context(
    command_buffer_markers: list[dict[str, Any]],
    barrier_markers: list[dict[str, Any]],
) -> dict[str, Any]:
    marker_totals = Counter(marker["kind"] for marker in command_buffer_markers + barrier_markers)
    cb_ids = sorted({int(marker["cb_id"]) for marker in command_buffer_markers if isinstance(marker.get("cb_id"), int)})
    return {
        "command_buffer_ids": cb_ids,
        "command_buffer_summary": [
            {"marker_type": marker_type, "count": count} for marker_type, count in sorted(marker_totals.items())
        ],
    }


def pair_command_buffer_spans(command_buffer_markers: list[dict[str, Any]]) -> dict[str, Any]:
    starts_by_cb: dict[int, list[dict[str, Any]]] = {}
    spans: list[dict[str, Any]] = []
    for marker in sorted(command_buffer_markers, key=_marker_position):
        kind = marker.get("kind")
        cb_id = marker.get("cb_id")
        if not isinstance(cb_id, int):
            continue
        if kind == "CB_START":
            starts_by_cb.setdefault(cb_id, []).append(marker)
        elif kind == "CB_END":
            start_list = starts_by_cb.get(cb_id)
            start_marker = start_list.pop() if start_list else None
            spans.append(
                {
                    "cb_id": cb_id,
                    "matched": start_marker is not None,
                    "begin_marker": start_marker,
                    "end_marker": marker,
                }
            )
    unmatched = [
        {"cb_id": cb_id, "marker": marker}
        for cb_id, markers in starts_by_cb.items()
        for marker in markers
    ]
    return {
        "command_buffer_spans": spans,
        "unmatched_command_buffer_begins": unmatched,
    }


def pair_barrier_spans(barrier_markers: list[dict[str, Any]]) -> dict[str, Any]:
    starts_by_cb: dict[int | None, list[dict[str, Any]]] = {}
    spans: list[dict[str, Any]] = []
    for marker in sorted(barrier_markers, key=_marker_position):
        kind = marker.get("kind")
        cb_id = marker.get("cb_id") if isinstance(marker.get("cb_id"), int) else None
        if kind == "BARRIER_START":
            starts_by_cb.setdefault(cb_id, []).append(marker)
        elif kind == "BARRIER_END":
            start_list = starts_by_cb.get(cb_id)
            start_marker = start_list.pop() if start_list else None
            spans.append(
                {
                    "cb_id": cb_id,
                    "matched": start_marker is not None,
                    "begin_marker": start_marker,
                    "end_marker": marker,
                }
            )
    unmatched = [
        {"cb_id": cb_id, "marker": marker}
        for cb_id, markers in starts_by_cb.items()
        for marker in markers
    ]
    return {
        "barrier_spans": spans,
        "unmatched_barrier_begins": unmatched,
    }
