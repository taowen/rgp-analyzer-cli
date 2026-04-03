from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .common import resolve_tinygrad_path, tinygrad_import_path


def analyze_tinygrad_sqtt(
    report: dict[str, Any],
    *,
    tinygrad_path: Path | None = None,
    stream_limit: int = 1,
    packet_limit: int = 20000,
    preview_limit: int = 40,
) -> dict[str, Any]:
    resolved = resolve_tinygrad_path(tinygrad_path)

    with tinygrad_import_path(resolved):
        from tinygrad.renderer.amd.sqtt import decode, format_packet

        streams: list[dict[str, Any]] = []
        count = len(report["sqtt_data_chunks"])
        if stream_limit > 0:
            count = min(count, stream_limit)
        for index in range(count):
            chunk = report["sqtt_data_chunks"][index]
            desc = report["sqtt_descs"][index] if index < len(report["sqtt_descs"]) else {}
            blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]
            type_counts: Counter[str] = Counter()
            event_counts: Counter[str] = Counter()
            preview: list[dict[str, Any]] = []
            packet_count = 0
            last_packet_type = None
            first_time = None
            last_time = None
            wave_start_count = 0
            wave_end_count = 0
            alu_exec_count = 0
            vmem_exec_count = 0
            wait_ready_count = 0
            for packet in decode(blob):
                packet_count += 1
                packet_type = type(packet).__name__
                type_counts[packet_type] += 1
                packet_time = getattr(packet, "_time", None)
                if packet_time is not None:
                    if first_time is None:
                        first_time = packet_time
                    last_time = packet_time
                if packet_type in {"WAVESTART", "WAVESTART_RDNA4", "CDNA_WAVESTART"}:
                    wave_start_count += 1
                elif packet_type in {"WAVEEND", "CDNA_WAVEEND"}:
                    wave_end_count += 1
                elif packet_type == "ALUEXEC":
                    alu_exec_count += 1
                elif packet_type == "VMEMEXEC":
                    vmem_exec_count += 1
                elif packet_type == "WAVERDY":
                    wait_ready_count += 1
                if packet_type in {"EVENT", "EVENT_BIG"}:
                    event_value = getattr(packet, "event", None)
                    if event_value is not None:
                        event_counts[f"event_{event_value}"] += 1
                if len(preview) < preview_limit:
                    preview.append(
                        {
                            "time": packet_time,
                            "type": packet_type,
                            "text": format_packet(packet),
                        }
                    )
                last_packet_type = packet_type
                if packet_limit > 0 and packet_count >= packet_limit:
                    break
            streams.append(
                {
                    "stream_index": index,
                    "shader_engine_index": desc.get("shader_engine_index"),
                    "compute_unit_index": desc.get("compute_unit_index"),
                    "packet_count": packet_count,
                    "truncated": packet_limit > 0 and packet_count >= packet_limit,
                    "last_packet_type": last_packet_type,
                    "first_time": first_time,
                    "last_time": last_time,
                    "wave_start_count": wave_start_count,
                    "wave_end_count": wave_end_count,
                    "alu_exec_count": alu_exec_count,
                    "vmem_exec_count": vmem_exec_count,
                    "wave_ready_count": wait_ready_count,
                    "type_counts": [
                        {"packet_type": packet_type, "count": count}
                        for packet_type, count in type_counts.most_common()
                    ],
                    "event_counts": [
                        {"event": event, "count": count}
                        for event, count in event_counts.most_common(16)
                    ],
                    "preview": preview,
                }
            )
    return {"tinygrad_path": str(resolved), "stream_count": len(streams), "streams": streams}


def render_tinygrad_sqtt(result: dict[str, Any], limit: int = 20, preview_limit: int = 20) -> str:
    lines = [f"tinygrad_sqtt: path={result['tinygrad_path']}"]
    for stream in result.get("streams", []):
        lines.append(
            f"  stream[{stream['stream_index']}] se={stream.get('shader_engine_index')} cu={stream.get('compute_unit_index')} "
            f"packets={stream['packet_count']} truncated={int(stream['truncated'])} last={stream.get('last_packet_type')} "
            f"time_range={stream.get('first_time')}..{stream.get('last_time')}"
        )
        lines.append(
            f"    waves start={stream.get('wave_start_count')} end={stream.get('wave_end_count')} "
            f"aluexec={stream.get('alu_exec_count')} vmemexec={stream.get('vmem_exec_count')} "
            f"waverdy={stream.get('wave_ready_count')}"
        )
        for item in stream.get("type_counts", [])[:limit]:
            lines.append(f"    {item['packet_type']} count={item['count']}")
        for item in stream.get("event_counts", [])[: min(limit, 8)]:
            lines.append(f"    {item['event']} count={item['count']}")
        for item in stream.get("preview", [])[:preview_limit]:
            lines.append(f"    {item['text']}")
    return "\n".join(lines)
