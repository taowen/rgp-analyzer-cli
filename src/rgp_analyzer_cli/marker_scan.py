from __future__ import annotations

import struct
from collections import Counter
from typing import Any


MARKER_IDENTIFIER_NAMES = {
    0: "EVENT",
    1: "CB_START",
    2: "CB_END",
    3: "BARRIER_START",
    4: "BARRIER_END",
    5: "USER_EVENT",
    6: "GENERAL_API",
    7: "SYNC",
    8: "PRESENT",
    9: "LAYOUT_TRANSITION",
    10: "RENDER_PASS",
    12: "BIND_PIPELINE",
}

MARKER_EXPECTED_EXT_DWORDS = {
    0: {2, 5},
    1: {3},
    2: {2},
    3: {1},
    4: {1},
    5: {0, 1},
    6: {0},
    9: {1},
    12: {2},
}

MARKER_FIXED_SIZE_DWORDS = {
    0: {3, 6},
    1: {4},
    2: {3},
    3: {2},
    4: {2},
    6: {1},
    9: {2},
    12: {3},
}

GENERAL_API_NAMES = {
    0: "ApiCmdBindPipeline",
    1: "ApiCmdBindDescriptorSets",
    2: "ApiCmdBindIndexBuffer",
    3: "ApiCmdBindVertexBuffers",
    4: "ApiCmdDraw",
    5: "ApiCmdDrawIndexed",
    6: "ApiCmdDrawIndirect",
    7: "ApiCmdDrawIndexedIndirect",
    8: "ApiCmdDrawIndirectCountAMD",
    9: "ApiCmdDrawIndexedIndirectCountAMD",
    10: "ApiCmdDispatch",
    11: "ApiCmdDispatchIndirect",
    12: "ApiCmdCopyBuffer",
    13: "ApiCmdCopyImage",
    14: "ApiCmdBlitImage",
    15: "ApiCmdCopyBufferToImage",
    16: "ApiCmdCopyImageToBuffer",
    17: "ApiCmdUpdateBuffer",
    18: "ApiCmdFillBuffer",
    19: "ApiCmdClearColorImage",
    20: "ApiCmdClearDepthStencilImage",
    21: "ApiCmdClearAttachments",
    22: "ApiCmdResolveImage",
    23: "ApiCmdWaitEvents",
    24: "ApiCmdPipelineBarrier",
    25: "ApiCmdBeginQuery",
    26: "ApiCmdEndQuery",
    27: "ApiCmdResetQueryPool",
    28: "ApiCmdWriteTimestamp",
    29: "ApiCmdCopyQueryPoolResults",
    30: "ApiCmdPushConstants",
    31: "ApiCmdBeginRenderPass",
    32: "ApiCmdNextSubpass",
    33: "ApiCmdEndRenderPass",
    34: "ApiCmdExecuteCommands",
    35: "ApiCmdSetViewport",
    36: "ApiCmdSetScissor",
    37: "ApiCmdSetLineWidth",
    38: "ApiCmdSetDepthBias",
    39: "ApiCmdSetBlendConstants",
    40: "ApiCmdSetDepthBounds",
    41: "ApiCmdSetStencilCompareMask",
    42: "ApiCmdSetStencilWriteMask",
    43: "ApiCmdSetStencilReference",
    44: "ApiCmdDrawIndirectCount",
    45: "ApiCmdDrawIndexedIndirectCount",
    47: "ApiCmdDrawMeshTasksEXT",
    48: "ApiCmdDrawMeshTasksIndirectCountEXT",
    49: "ApiCmdDrawMeshTasksIndirectEXT",
}

EVENT_TYPE_NAMES = {
    0: "EventCmdDraw",
    1: "EventCmdDrawIndexed",
    2: "EventCmdDrawIndirect",
    3: "EventCmdDrawIndexedIndirect",
    4: "EventCmdDrawIndirectCountAMD",
    5: "EventCmdDrawIndexedIndirectCountAMD",
    6: "EventCmdDispatch",
    7: "EventCmdDispatchIndirect",
    8: "EventCmdCopyBuffer",
    9: "EventCmdCopyImage",
    10: "EventCmdBlitImage",
    11: "EventCmdCopyBufferToImage",
    12: "EventCmdCopyImageToBuffer",
    13: "EventCmdUpdateBuffer",
    14: "EventCmdFillBuffer",
    15: "EventCmdClearColorImage",
    16: "EventCmdClearDepthStencilImage",
    17: "EventCmdClearAttachments",
    18: "EventCmdResolveImage",
    19: "EventCmdWaitEvents",
    20: "EventCmdPipelineBarrier",
    21: "EventCmdResetQueryPool",
    22: "EventCmdCopyQueryPoolResults",
    23: "EventRenderPassColorClear",
    24: "EventRenderPassDepthStencilClear",
    25: "EventRenderPassResolve",
    26: "EventInternalUnknown",
    27: "EventCmdDrawIndirectCount",
    28: "EventCmdDrawIndexedIndirectCount",
    30: "EventCmdTraceRaysKHR",
    31: "EventCmdTraceRaysIndirectKHR",
    32: "EventCmdBuildAccelerationStructuresKHR",
    33: "EventCmdBuildAccelerationStructuresIndirectKHR",
    34: "EventCmdCopyAccelerationStructureKHR",
    35: "EventCmdCopyAccelerationStructureToMemoryKHR",
    36: "EventCmdCopyMemoryToAccelerationStructureKHR",
    41: "EventCmdDrawMeshTasksEXT",
    42: "EventCmdDrawMeshTasksIndirectCountEXT",
    43: "EventCmdDrawMeshTasksIndirectEXT",
}

USER_EVENT_TYPE_NAMES = {
    0: "UserEventTrigger",
    1: "UserEventPop",
    2: "UserEventPush",
    3: "UserEventObjectName",
}

HIGH_CONFIDENCE_APIS = {
    "ApiCmdDispatch",
    "ApiCmdDispatchIndirect",
    "ApiCmdPushConstants",
    "ApiCmdWriteTimestamp",
    "ApiCmdPipelineBarrier",
    "ApiCmdCopyBuffer",
    "ApiCmdCopyImage",
    "ApiCmdCopyBufferToImage",
    "ApiCmdCopyImageToBuffer",
    "ApiCmdFillBuffer",
    "ApiCmdUpdateBuffer",
}


def _dwords_for_stream(report: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    chunk = report["sqtt_data_chunks"][index]
    desc = report["sqtt_descs"][index] if index < len(report["sqtt_descs"]) else {}
    blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]
    dwords = struct.unpack_from(f"<{len(blob) // 4}I", blob, 0)
    return chunk, desc, dwords


def _base_marker(stream_index: int, dword_index: int, dwords: tuple[int, ...], identifier: int, ext_dwords: int) -> dict[str, Any]:
    size_dwords = ext_dwords + 1
    words = list(dwords[dword_index : dword_index + size_dwords])
    return {
        "stream_index": stream_index,
        "dword_index": dword_index,
        "byte_offset": dword_index * 4,
        "identifier": identifier,
        "identifier_name": MARKER_IDENTIFIER_NAMES.get(identifier, f"UNKNOWN_{identifier}"),
        "ext_dwords": ext_dwords,
        "size_dwords": size_dwords,
        "raw": words[0] if words else 0,
        "dwords": words,
    }


def _decode_general_api(marker: dict[str, Any]) -> dict[str, Any]:
    value = marker["raw"]
    api_type = (value >> 7) & 0xFFFFF
    api_name = GENERAL_API_NAMES.get(api_type)
    phase = "end" if ((value >> 27) & 0x1) else "begin"
    confidence = "high" if api_name in HIGH_CONFIDENCE_APIS else "low"
    marker.update(
        {
            "api_type": api_type,
            "api_name": api_name or f"ApiUnknown_{api_type}",
            "phase": phase,
            "confidence": confidence,
        }
    )
    return marker


def _decode_event(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    api_type = (value >> 7) & 0xFFFFFF
    marker.update(
        {
            "event_type": api_type,
            "event_name": EVENT_TYPE_NAMES.get(api_type, f"EventUnknown_{api_type}"),
            "has_thread_dims": bool((value >> 31) & 0x1),
            "cb_id": words[1] & 0xFFFFF if len(words) > 1 else 0,
            "cmd_id": words[2] if len(words) > 2 else 0,
        }
    )
    if len(words) >= 6:
        marker["thread_dims"] = {"x": words[3], "y": words[4], "z": words[5]}
    return marker


def _decode_barrier_start(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    reason_word = words[1] if len(words) > 1 else 0
    marker.update(
        {
            "cb_id": (value >> 7) & 0xFFFFF,
            "driver_reason": reason_word & 0x7FFFFFFF,
            "internal": bool((reason_word >> 31) & 0x1),
        }
    )
    return marker


def _decode_barrier_end(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    flags = words[1] if len(words) > 1 else 0
    marker.update(
        {
            "cb_id": (value >> 7) & 0xFFFFF,
            "wait_on_eop_ts": bool((value >> 27) & 0x1),
            "vs_partial_flush": bool((value >> 28) & 0x1),
            "ps_partial_flush": bool((value >> 29) & 0x1),
            "cs_partial_flush": bool((value >> 30) & 0x1),
            "pfp_sync_me": bool((value >> 31) & 0x1),
            "sync_cp_dma": bool(flags & (1 << 0)),
            "inval_tcp": bool(flags & (1 << 1)),
            "inval_sqI": bool(flags & (1 << 2)),
            "inval_sqK": bool(flags & (1 << 3)),
            "flush_tcc": bool(flags & (1 << 4)),
            "inval_tcc": bool(flags & (1 << 5)),
            "flush_cb": bool(flags & (1 << 6)),
            "inval_cb": bool(flags & (1 << 7)),
            "flush_db": bool(flags & (1 << 8)),
            "inval_db": bool(flags & (1 << 9)),
            "num_layout_transitions": (flags >> 10) & 0xFFFF,
            "inval_gl1": bool(flags & (1 << 30)),
            "wait_on_ts": bool(flags & (1 << 31)),
        }
    )
    return marker


def _decode_layout_transition(marker: dict[str, Any]) -> dict[str, Any]:
    value = marker["raw"]
    marker.update(
        {
            "depth_stencil_expand": bool((value >> 7) & 0x1),
            "htile_hiz_range_expand": bool((value >> 8) & 0x1),
            "depth_stencil_resummarize": bool((value >> 9) & 0x1),
            "dcc_decompress": bool((value >> 10) & 0x1),
            "fmask_decompress": bool((value >> 11) & 0x1),
            "fast_clear_eliminate": bool((value >> 12) & 0x1),
            "fmask_color_expand": bool((value >> 13) & 0x1),
            "init_mask_ram": bool((value >> 14) & 0x1),
        }
    )
    return marker


def _decode_user_event(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    data_type = (value >> 12) & 0xFF
    marker.update(
        {
            "data_type": data_type,
            "data_type_name": USER_EVENT_TYPE_NAMES.get(data_type, f"UserEventUnknown_{data_type}"),
        }
    )
    if len(words) > 1:
        marker["length"] = words[1]
    return marker


def _decode_pipeline_bind(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    api_pso_hash = 0
    if len(words) >= 3:
        api_pso_hash = words[1] | (words[2] << 32)
    marker.update(
        {
            "bind_point": "compute" if ((value >> 7) & 0x1) else "graphics",
            "cb_id": (value >> 8) & 0xFFFFF,
            "api_pso_hash": api_pso_hash,
        }
    )
    return marker


def _decode_cb_start(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    device_id = 0
    if len(words) >= 3:
        device_id = words[1] | (words[2] << 32)
    marker.update(
        {
            "cb_id": (value >> 7) & 0xFFFFF,
            "queue": (value >> 27) & 0x1F,
            "device_id": device_id,
            "queue_flags": words[3] if len(words) > 3 else 0,
        }
    )
    return marker


def _decode_cb_end(marker: dict[str, Any]) -> dict[str, Any]:
    words = marker["dwords"]
    value = words[0]
    device_id = 0
    if len(words) >= 3:
        device_id = words[1] | (words[2] << 32)
    marker.update({"cb_id": (value >> 7) & 0xFFFFF, "device_id": device_id})
    return marker


def _decode_marker(marker: dict[str, Any]) -> dict[str, Any]:
    identifier = marker["identifier"]
    if identifier == 0:
        return _decode_event(marker)
    if identifier == 1:
        return _decode_cb_start(marker)
    if identifier == 2:
        return _decode_cb_end(marker)
    if identifier == 3:
        return _decode_barrier_start(marker)
    if identifier == 4:
        return _decode_barrier_end(marker)
    if identifier == 5:
        return _decode_user_event(marker)
    if identifier == 6:
        return _decode_general_api(marker)
    if identifier == 9:
        return _decode_layout_transition(marker)
    if identifier == 12:
        return _decode_pipeline_bind(marker)
    return marker


def _known_api_pso_hashes(report: dict[str, Any]) -> set[int]:
    hashes: set[int] = set()
    for chunk in report.get("pso_correlations", []):
        for record in chunk.get("records", []):
            api_pso_hash = record.get("api_pso_hash")
            if isinstance(api_pso_hash, int):
                hashes.add(api_pso_hash)
    return hashes


def _assign_marker_confidence(report: dict[str, Any], marker: dict[str, Any], known_pso_hashes: set[int]) -> dict[str, Any]:
    confidence = "low"
    if marker["identifier"] == 6:
        confidence = marker.get("confidence", "low")
    elif marker["identifier"] == 12:
        confidence = "high" if marker.get("api_pso_hash") in known_pso_hashes else "low"
    elif marker["identifier"] == 0:
        event_name = marker.get("event_name", "")
        dims = marker.get("thread_dims")
        if event_name in {"EventCmdDispatch", "EventCmdDispatchIndirect"} and dims:
            x = int(dims.get("x", 0))
            y = int(dims.get("y", 0))
            z = int(dims.get("z", 0))
            if 0 < x <= 1_048_576 and 0 < y <= 1_048_576 and 0 < z <= 1_048_576:
                confidence = "high"
        elif event_name in {"EventCmdPipelineBarrier", "EventCmdCopyBuffer", "EventCmdUpdateBuffer"}:
            confidence = "medium"
    elif marker["identifier"] == 1:
        if marker.get("cb_id") and marker.get("device_id"):
            confidence = "high"
    elif marker["identifier"] == 2:
        if marker.get("cb_id") and marker.get("device_id"):
            confidence = "high"
    elif marker["identifier"] in {3, 4}:
        if marker.get("cb_id"):
            confidence = "medium"
    marker["confidence"] = confidence
    return marker


def scan_markers(report: dict[str, Any], stream_limit: int = 0) -> dict[str, Any]:
    count = len(report["sqtt_data_chunks"])
    if stream_limit and stream_limit > 0:
        count = min(count, stream_limit)

    known_pso_hashes = _known_api_pso_hashes(report)
    streams: list[dict[str, Any]] = []
    for index in range(count):
        _, desc, dwords = _dwords_for_stream(report, index)
        markers: list[dict[str, Any]] = []
        summary_counter: Counter[tuple[str, str, str]] = Counter()

        for dword_index, value in enumerate(dwords):
            identifier = value & 0xF
            ext_dwords = (value >> 4) & 0x7
            expected = MARKER_EXPECTED_EXT_DWORDS.get(identifier)
            if expected is None or ext_dwords not in expected:
                continue
            size_dwords = ext_dwords + 1
            fixed_sizes = MARKER_FIXED_SIZE_DWORDS.get(identifier)
            if fixed_sizes is not None and size_dwords not in fixed_sizes:
                continue
            if dword_index + size_dwords > len(dwords):
                continue
            marker = _decode_marker(_base_marker(index, dword_index, dwords, identifier, ext_dwords))
            marker = _assign_marker_confidence(report, marker, known_pso_hashes)
            markers.append(marker)
            summary_counter[(marker["identifier_name"], str(ext_dwords), marker["confidence"])] += 1

        streams.append(
            {
                "stream_index": index,
                "shader_engine_index": desc.get("shader_engine_index"),
                "compute_unit_index": desc.get("compute_unit_index"),
                "marker_count": len(markers),
                "markers": markers,
                "summary": [
                    {"marker_type": marker_type, "ext_dwords": ext_dwords, "confidence": confidence, "count": count}
                    for (marker_type, ext_dwords, confidence), count in sorted(summary_counter.items())
                ],
            }
        )

    return {"stream_count": len(streams), "streams": streams}


def filter_markers(result: dict[str, Any], *, confidence: str | None = None) -> dict[str, Any]:
    if confidence is None:
        return result
    streams: list[dict[str, Any]] = []
    for stream in result.get("streams", []):
        markers = [marker for marker in stream.get("markers", []) if marker.get("confidence") == confidence]
        summary = [item for item in stream.get("summary", []) if item.get("confidence") == confidence]
        streams.append(
            {
                "stream_index": stream["stream_index"],
                "shader_engine_index": stream.get("shader_engine_index"),
                "compute_unit_index": stream.get("compute_unit_index"),
                "marker_count": len(markers),
                "markers": markers,
                "summary": summary,
            }
        )
    return {"stream_count": len(streams), "streams": streams}


def scan_general_api_markers(report: dict[str, Any], stream_limit: int = 0) -> dict[str, Any]:
    result = scan_markers(report, stream_limit=stream_limit)
    streams: list[dict[str, Any]] = []
    for stream in result["streams"]:
        markers = [marker for marker in stream["markers"] if marker["identifier"] == 6]
        summary_counter: Counter[tuple[str, str, str]] = Counter()
        for marker in markers:
            summary_counter[(marker["api_name"], marker["phase"], marker["confidence"])] += 1
        streams.append(
            {
                "stream_index": stream["stream_index"],
                "shader_engine_index": stream.get("shader_engine_index"),
                "compute_unit_index": stream.get("compute_unit_index"),
                "marker_count": len(markers),
                "markers": markers,
                "summary": [
                    {"api_name": api_name, "phase": phase, "confidence": confidence, "count": count}
                    for (api_name, phase, confidence), count in sorted(summary_counter.items())
                ],
            }
        )
    return {"stream_count": len(streams), "streams": streams}


def render_markers(result: dict[str, Any], limit: int = 20) -> str:
    lines = ["markers:"]
    for stream in result.get("streams", []):
        lines.append(
            f"  stream[{stream['stream_index']}] se={stream.get('shader_engine_index')} cu={stream.get('compute_unit_index')} "
            f"markers={stream['marker_count']}"
        )
        for item in stream.get("summary", [])[:limit]:
            lines.append(
                f"    {item['marker_type']} ext_dwords={item['ext_dwords']} confidence={item['confidence']} count={item['count']}"
            )
        for marker in stream.get("markers", [])[:limit]:
            detail = marker["identifier_name"]
            if marker["identifier"] == 6:
                detail = f"{marker['api_name']} {marker['phase']} confidence={marker['confidence']}"
            elif marker["identifier"] == 0:
                dims = marker.get("thread_dims")
                dims_text = f" dims={dims['x']}x{dims['y']}x{dims['z']}" if dims else ""
                detail = f"{marker['event_name']} cmd_id={marker['cmd_id']}{dims_text}"
            elif marker["identifier"] == 12:
                detail = f"{marker['identifier_name']} bind_point={marker['bind_point']} api_pso_hash={marker['api_pso_hash']}"
            elif marker["identifier"] == 4:
                detail = (
                    f"{marker['identifier_name']} cs_partial_flush={int(marker['cs_partial_flush'])} "
                    f"wait_on_ts={int(marker['wait_on_ts'])} num_layout_transitions={marker['num_layout_transitions']}"
                )
            elif marker["identifier"] == 3:
                detail = (
                    f"{marker['identifier_name']} driver_reason={marker['driver_reason']} internal={int(marker['internal'])}"
                )
            elif marker["identifier"] == 5:
                detail = f"{marker['identifier_name']} {marker['data_type_name']} length={marker.get('length', 0)}"
            elif marker["identifier"] == 9:
                flags = [name for name in (
                    "depth_stencil_expand",
                    "htile_hiz_range_expand",
                    "depth_stencil_resummarize",
                    "dcc_decompress",
                    "fmask_decompress",
                    "fast_clear_eliminate",
                    "fmask_color_expand",
                    "init_mask_ram",
                ) if marker.get(name)]
                detail = f"{marker['identifier_name']} flags={','.join(flags) if flags else 'none'}"
            lines.append(
                f"    at_dword={marker['dword_index']} byte_offset={marker['byte_offset']} confidence={marker['confidence']} "
                f"{detail} raw=0x{marker['raw']:08x}"
            )
    return "\n".join(lines)


def render_general_api_markers(result: dict[str, Any], limit: int = 20) -> str:
    lines = ["general_api_markers:"]
    for stream in result.get("streams", []):
        lines.append(
            f"  stream[{stream['stream_index']}] se={stream.get('shader_engine_index')} cu={stream.get('compute_unit_index')} "
            f"markers={stream['marker_count']}"
        )
        for item in stream.get("summary", [])[:limit]:
            lines.append(
                f"    {item['api_name']} {item['phase']} confidence={item['confidence']} count={item['count']}"
            )
        for marker in stream.get("markers", [])[:limit]:
            lines.append(
                f"    at_dword={marker['dword_index']} byte_offset={marker['byte_offset']} "
                f"{marker['api_name']} {marker['phase']} confidence={marker['confidence']} raw=0x{marker['raw']:08x}"
            )
    return "\n".join(lines)


def summarize_general_api_markers(result: dict[str, Any]) -> dict[str, Any]:
    totals: Counter[tuple[str, str, str]] = Counter()
    for stream in result.get("streams", []):
        for item in stream.get("summary", []):
            totals[(item["api_name"], item["phase"], item["confidence"])] += int(item["count"])
    ordered = [
        {"api_name": api_name, "phase": phase, "confidence": confidence, "count": count}
        for (api_name, phase, confidence), count in sorted(
            totals.items(), key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1])
        )
    ]
    high_confidence = [item for item in ordered if item["confidence"] == "high"]
    low_confidence = [item for item in ordered if item["confidence"] == "low"]
    return {"totals": ordered, "high_confidence": high_confidence, "low_confidence": low_confidence}


def summarize_markers(result: dict[str, Any]) -> dict[str, Any]:
    totals: Counter[tuple[str, str]] = Counter()
    for stream in result.get("streams", []):
        for item in stream.get("summary", []):
            totals[(item["marker_type"], item["confidence"])] += int(item["count"])
    ordered = [
        {"marker_type": marker_type, "confidence": confidence, "count": count}
        for (marker_type, confidence), count in sorted(totals.items())
    ]
    high_confidence = [item for item in ordered if item["confidence"] == "high"]
    return {"totals": ordered, "high_confidence": high_confidence}
