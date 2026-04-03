from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..marker_scan import (
    MARKER_FIXED_SIZE_DWORDS,
    MARKER_IDENTIFIER_NAMES,
    MARKER_EXPECTED_EXT_DWORDS,
    _assign_marker_confidence,
    _decode_marker,
    _known_api_pso_hashes,
    scan_general_api_markers,
)
from .common import resolve_tinygrad_path, tinygrad_import_path

ROCPROF_INSTRUMENT_ENABLE_MAGIC = 0x434F5200


def _marker_position_key(marker: dict[str, Any]) -> tuple[int, int]:
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


def reconstruct_tinygrad_userdata_markers(
    report: dict[str, Any],
    *,
    tinygrad_path: Path | None = None,
    stream_limit: int = 0,
) -> dict[str, Any]:
    resolved = resolve_tinygrad_path(tinygrad_path)
    general_api = scan_general_api_markers(report, stream_limit=stream_limit)
    general_raws_by_stream = {
        stream["stream_index"]: {int(marker["raw"]) for marker in stream.get("markers", []) if marker.get("confidence") == "high"}
        for stream in general_api.get("streams", [])
    }
    known_pso_hashes = _known_api_pso_hashes(report)
    count = len(report["sqtt_data_chunks"])
    if stream_limit and stream_limit > 0:
        count = min(count, stream_limit)

    with tinygrad_import_path(resolved):
        from tinygrad.renderer.amd.sqtt import decode

        raw_reg_packets: list[list[dict[str, Any]]] = []
        signature_counts: Counter[tuple[int, int]] = Counter()
        rocprof_stream_summaries: list[dict[str, Any]] = []
        for index in range(count):
            chunk = report["sqtt_data_chunks"][index]
            blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]
            stream_raws = general_raws_by_stream.get(index, set())
            reg_packets: list[dict[str, Any]] = []
            rocprof_enable_signatures: Counter[tuple[int, int]] = Counter()
            rocprof_headers: list[dict[str, Any]] = []
            for packet_index, packet in enumerate(decode(blob)):
                if type(packet).__name__ != "REG":
                    continue
                reg = {
                    "packet_index": packet_index,
                    "time": getattr(packet, "_time", None),
                    "slot": int(getattr(packet, "slot", 0)),
                    "hi_byte": int(getattr(packet, "hi_byte", 0)),
                    "subop": int(getattr(packet, "subop", 0)),
                    "val32": int(getattr(packet, "val32", 0)),
                }
                reg_packets.append(reg)
                if reg["val32"] in stream_raws:
                    signature_counts[(reg["hi_byte"], reg["subop"])] += 1
                if reg["val32"] == ROCPROF_INSTRUMENT_ENABLE_MAGIC:
                    rocprof_enable_signatures[(reg["hi_byte"], reg["subop"])] += 1
            raw_reg_packets.append(reg_packets)
            enabled_signature_set = set(rocprof_enable_signatures.keys())
            if enabled_signature_set:
                for reg in reg_packets:
                    signature = (reg["hi_byte"], reg["subop"])
                    if signature not in enabled_signature_set:
                        continue
                    opcode = reg["val32"] & 0xFF
                    if opcode not in {4, 5, 6}:
                        continue
                    rocprof_headers.append(
                        {
                            "packet_index": reg["packet_index"],
                            "hi_byte": reg["hi_byte"],
                            "subop": reg["subop"],
                            "opcode": opcode,
                            "type": (reg["val32"] >> 8) & 0xF,
                            "raw": reg["val32"],
                        }
                    )
            rocprof_stream_summaries.append(
                {
                    "enable_packet_count": sum(rocprof_enable_signatures.values()),
                    "enable_signatures": [
                        {"hi_byte": hi_byte, "subop": subop, "count": count}
                        for (hi_byte, subop), count in sorted(rocprof_enable_signatures.items())
                    ],
                    "header_packet_count": len(rocprof_headers),
                    "header_packets": rocprof_headers[:16],
                }
            )

    inferred_signature = None
    if signature_counts:
        hi_byte, subop = signature_counts.most_common(1)[0][0]
        sibling_subops = sorted(
            {
                candidate_subop
                for candidate_hi_byte, candidate_subop in signature_counts
                if candidate_hi_byte == hi_byte and abs(candidate_subop - subop) <= 1
            }
            | {subop, subop + 1}
        )
        inferred_signature = {
            "hi_byte": hi_byte,
            "subop": subop,
            "subops": sibling_subops,
            "match_count": sum(signature_counts.get((hi_byte, candidate), 0) for candidate in sibling_subops),
        }

    streams: list[dict[str, Any]] = []
    for index in range(count):
        desc = report["sqtt_descs"][index] if index < len(report["sqtt_descs"]) else {}
        reg_packets = raw_reg_packets[index]
        if inferred_signature is None:
            streams.append(
                {
                    "stream_index": index,
                    "shader_engine_index": desc.get("shader_engine_index"),
                    "compute_unit_index": desc.get("compute_unit_index"),
                    "userdata_reg_packets": [],
                    "markers": [],
                    "summary": [],
                }
            )
            continue

        signature_packets = [
            packet
            for packet in reg_packets
            if packet["hi_byte"] == inferred_signature["hi_byte"] and packet["subop"] in set(inferred_signature["subops"])
        ]
        markers: list[dict[str, Any]] = []
        summary_counts: Counter[tuple[str, str]] = Counter()
        i = 0
        while i < len(signature_packets):
            first = signature_packets[i]
            raw = first["val32"]
            identifier = raw & 0xF
            ext_dwords = (raw >> 4) & 0x7
            fixed_sizes = MARKER_FIXED_SIZE_DWORDS.get(identifier)
            fixed_size = min(fixed_sizes) if fixed_sizes else None
            expected = MARKER_EXPECTED_EXT_DWORDS.get(identifier)
            if expected is not None and ext_dwords not in expected and identifier not in {1, 2, 3, 4, 6, 12}:
                i += 1
                continue
            if fixed_size is None and expected is None:
                i += 1
                continue
            size_dwords = fixed_size or (ext_dwords + 1)
            if i + size_dwords > len(signature_packets):
                i += 1
                continue
            candidate_packets = signature_packets[i : i + size_dwords]
            words = [packet["val32"] for packet in candidate_packets]
            if any(candidate_packets[j + 1]["packet_index"] - candidate_packets[j]["packet_index"] > 8 for j in range(len(candidate_packets) - 1)):
                i += 1
                continue
            marker = {
                "stream_index": index,
                "packet_index": first["packet_index"],
                "dword_index": first["packet_index"],
                "byte_offset": None,
                "identifier": identifier,
                "identifier_name": MARKER_IDENTIFIER_NAMES.get(identifier, f"UNKNOWN_{identifier}"),
                "ext_dwords": ext_dwords,
                "size_dwords": size_dwords,
                "raw": raw,
                "dwords": words,
                "source": "tinygrad_userdata_reg",
                "userdata_signature": [inferred_signature["hi_byte"], *inferred_signature["subops"]],
            }
            marker = _decode_marker(marker)
            marker = _assign_marker_confidence(report, marker, known_pso_hashes)
            markers.append(marker)
            summary_counts[(marker["identifier_name"], marker.get("confidence", "low"))] += 1
            i += size_dwords

        streams.append(
            {
                "stream_index": index,
                "shader_engine_index": desc.get("shader_engine_index"),
                "compute_unit_index": desc.get("compute_unit_index"),
                "userdata_reg_packets": signature_packets,
                "rocprof_instrumentation": rocprof_stream_summaries[index],
                "markers": sorted(markers, key=_marker_position_key),
                "summary": [
                    {"marker_type": marker_type, "confidence": confidence, "count": item_count}
                    for (marker_type, confidence), item_count in sorted(summary_counts.items())
                ],
            }
        )

    total_enable_packets = sum(item["enable_packet_count"] for item in rocprof_stream_summaries)
    streams_with_enable = sum(1 for item in rocprof_stream_summaries if item["enable_packet_count"] > 0)
    return {
        "tinygrad_path": str(resolved),
        "inferred_userdata_signature": inferred_signature,
        "rocprof_instrumentation": {
            "enable_magic": ROCPROF_INSTRUMENT_ENABLE_MAGIC,
            "total_enable_packets": total_enable_packets,
            "streams_with_enable": streams_with_enable,
        },
        "stream_count": len(streams),
        "streams": streams,
    }
