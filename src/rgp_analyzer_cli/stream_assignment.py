from __future__ import annotations

from typing import Any


def mark_entries_seen_in_bind_markers(entries: list[dict[str, Any]], stream_markers: list[dict[str, Any]]) -> None:
    bind_hashes = {
        int(marker["api_pso_hash"])
        for stream in stream_markers
        for marker in stream.get("bind_pipeline_markers", [])
        if isinstance(marker.get("api_pso_hash"), int)
    }
    for entry in entries:
        api_pso_hash = entry.get("api_pso_hash")
        entry["seen_in_bind_pipeline_markers"] = bool(api_pso_hash in bind_hashes if api_pso_hash is not None else False)


def assign_streams_to_entries(
    entries: list[dict[str, Any]],
    stream_markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved_entries = [entry for entry in entries if entry.get("resolved")]
    single_resolved = resolved_entries[0] if len(resolved_entries) == 1 else None

    for stream in stream_markers:
        stream_entry = None
        stream_match_kind = None
        bind_matches = [
            entry
            for entry in resolved_entries
            if any(marker.get("api_pso_hash") == entry.get("api_pso_hash") for marker in stream.get("bind_pipeline_markers", []))
        ]
        if len(bind_matches) == 1:
            stream_entry = bind_matches[0]
            stream_match_kind = "bind_pipeline_marker"
        elif single_resolved is not None and stream.get("dispatch_event_markers"):
            stream_entry = single_resolved
            stream_match_kind = "single_resolved_code_object_with_dispatch"

        stream["resolved_code_object_index"] = stream_entry.get("code_object_index") if stream_entry else None
        stream["resolved_api_pso_hash"] = stream_entry.get("api_pso_hash") if stream_entry else None
        stream["resolved_pipeline_hash"] = stream_entry.get("pipeline_hash") if stream_entry else None
        stream["stream_match_kind"] = stream_match_kind

    return stream_markers
