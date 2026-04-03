from __future__ import annotations

from collections import Counter
import struct
from typing import Any

from .decode_bridge import decoder_bridge


def chunk_counts(report: dict[str, Any]) -> dict[str, int]:
    counts = Counter(chunk["type_name"] for chunk in report["chunks"])
    return dict(sorted(counts.items()))


def flattened_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    queue_infos = []
    for queue_chunk in report["queue_event_chunks"]:
        queue_infos.extend(queue_chunk["queue_infos"])

    events = []
    for queue_chunk in report["queue_event_chunks"]:
        for event in queue_chunk["queue_events"]:
            enriched = dict(event)
            queue_index = event["queue_info_index"]
            if 0 <= queue_index < len(queue_infos):
                q = queue_infos[queue_index]
                enriched["queue_type_name"] = q["queue_type_name"]
                enriched["engine_type_name"] = q["engine_type_name"]
            else:
                enriched["queue_type_name"] = "UNKNOWN"
                enriched["engine_type_name"] = "UNKNOWN"
            events.append(enriched)
    return events


def code_object_summary(report: dict[str, Any]) -> dict[str, Any]:
    records = []
    for db in report["code_object_databases"]:
        records.extend(db["records"])

    total_payload = sum(record["payload_size"] for record in records)
    return {
        "database_count": len(report["code_object_databases"]),
        "record_count": len(records),
        "total_payload_bytes": total_payload,
        "elf_record_count": sum(1 for record in records if record.get("elf", {}).get("valid_elf")),
        "largest_records": sorted(records, key=lambda item: item["payload_size"], reverse=True)[:10],
    }
def pso_summary(report: dict[str, Any]) -> dict[str, Any]:
    records = []
    for chunk in report["pso_correlations"]:
        records.extend(chunk["records"])
    return {
        "chunk_count": len(report["pso_correlations"]),
        "record_count": len(records),
        "named_records": [record for record in records if record["api_level_obj_name"]],
    }


def sqtt_summary(report: dict[str, Any]) -> dict[str, Any]:
    total_trace_bytes = sum(chunk["size"] for chunk in report["sqtt_data_chunks"])
    return {
        "desc_count": len(report["sqtt_descs"]),
        "data_chunk_count": len(report["sqtt_data_chunks"]),
        "total_trace_bytes": total_trace_bytes,
    }


def event_summary(report: dict[str, Any]) -> dict[str, Any]:
    events = flattened_events(report)
    durations = [event["gpu_duration"] for event in events]
    return {
        "event_count": len(events),
        "total_gpu_duration": sum(durations),
        "max_gpu_duration": max(durations) if durations else 0,
        "min_gpu_duration": min(durations) if durations else 0,
        "avg_gpu_duration": (sum(durations) / len(durations)) if durations else 0.0,
        "compute_event_count": sum(1 for event in events if event["queue_type_name"] == "COMPUTE"),
        "queue_types": sorted({event["queue_type_name"] for event in events}),
    }


def correlation_summary(report: dict[str, Any]) -> dict[str, Any]:
    loader_records = []
    for chunk in report["loader_events"]:
        loader_records.extend(chunk["records"])

    pso_records = []
    for chunk in report["pso_correlations"]:
        pso_records.extend(chunk["records"])

    loader_hash_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in loader_records:
        key = tuple(record["code_object_hash"])
        loader_hash_map.setdefault(key, []).append(record)

    correlations = []
    unmatched_pso = []
    for record in pso_records:
        key = tuple(record["pipeline_hash"])
        matches = loader_hash_map.get(key, [])
        entry = {
            "pipeline_hash": list(key),
            "api_pso_hash": record["api_pso_hash"],
            "api_level_obj_name": record["api_level_obj_name"],
            "loader_match_count": len(matches),
            "loader_base_addresses": [match["base_address"] for match in matches],
            "loader_event_types": [match["loader_event_type_name"] for match in matches],
        }
        correlations.append(entry)
        if not matches:
            unmatched_pso.append(entry)

    return {
        "loader_record_count": len(loader_records),
        "pso_record_count": len(pso_records),
        "correlations": correlations,
        "matched_pso_count": len(correlations) - len(unmatched_pso),
        "unmatched_pso_count": len(unmatched_pso),
    }


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_events = event_summary(baseline)
    candidate_events = event_summary(candidate)
    baseline_code = code_object_summary(baseline)
    candidate_code = code_object_summary(candidate)
    baseline_sqtt = sqtt_summary(baseline)
    candidate_sqtt = sqtt_summary(candidate)

    def delta(new_value: float, old_value: float) -> dict[str, float | None]:
        absolute = new_value - old_value
        percent = None if old_value == 0 else (absolute / old_value) * 100.0
        return {"absolute": absolute, "percent": percent}

    return {
        "baseline_file": baseline["file"],
        "candidate_file": candidate["file"],
        "event_count": {
            "baseline": baseline_events["event_count"],
            "candidate": candidate_events["event_count"],
        },
        "total_gpu_duration": {
            "baseline": baseline_events["total_gpu_duration"],
            "candidate": candidate_events["total_gpu_duration"],
            "delta": delta(candidate_events["total_gpu_duration"], baseline_events["total_gpu_duration"]),
        },
        "max_gpu_duration": {
            "baseline": baseline_events["max_gpu_duration"],
            "candidate": candidate_events["max_gpu_duration"],
            "delta": delta(candidate_events["max_gpu_duration"], baseline_events["max_gpu_duration"]),
        },
        "avg_gpu_duration": {
            "baseline": baseline_events["avg_gpu_duration"],
            "candidate": candidate_events["avg_gpu_duration"],
            "delta": delta(candidate_events["avg_gpu_duration"], baseline_events["avg_gpu_duration"]),
        },
        "code_object_records": {
            "baseline": baseline_code["record_count"],
            "candidate": candidate_code["record_count"],
            "delta": candidate_code["record_count"] - baseline_code["record_count"],
        },
        "sqtt_trace_bytes": {
            "baseline": baseline_sqtt["total_trace_bytes"],
            "candidate": candidate_sqtt["total_trace_bytes"],
            "delta": delta(candidate_sqtt["total_trace_bytes"], baseline_sqtt["total_trace_bytes"]),
        },
        "chunk_counts": {
            "baseline": chunk_counts(baseline),
            "candidate": chunk_counts(candidate),
        },
        "queue_types": {
            "baseline": baseline_events["queue_types"],
            "candidate": candidate_events["queue_types"],
        },
    }


def scan_sqtt_payload(report: dict[str, Any], index: int, dword_limit: int = 64) -> dict[str, Any]:
    chunk = report["sqtt_data_chunks"][index]
    blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]
    dword_count = len(blob) // 4
    preview_count = min(dword_count, dword_limit)
    values = list(struct.unpack_from(f"<{preview_count}I", blob, 0)) if preview_count else []
    high_nibbles = Counter((value >> 28) & 0xF for value in values)
    return {
        "index": index,
        "size_bytes": len(blob),
        "dword_count": dword_count,
        "preview_dwords": values,
        "high_nibble_histogram": dict(sorted(high_nibbles.items())),
    }


def summarize_isa_text(disassembly: str) -> dict[str, Any]:
    opcodes: list[str] = []
    for line in disassembly.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        if ":" not in stripped:
            continue
        if "//" in stripped:
            stripped = stripped.split("//", 1)[0].rstrip()
        parts = stripped.split()
        if not parts:
            continue
        opcode = parts[0]
        if opcode.startswith(("v_", "s_", "buffer_", "flat_", "ds_", "global_", "image_", "exp", "tbuffer_")):
            opcodes.append(opcode)

    counts = Counter(opcodes)
    total = sum(counts.values())

    def category_count(prefixes: tuple[str, ...]) -> int:
        return sum(count for opcode, count in counts.items() if opcode.startswith(prefixes))

    return {
        "instruction_count": total,
        "unique_opcodes": len(counts),
        "opcode_counts": dict(counts.most_common(50)),
        "vector_alu": category_count(("v_",)),
        "scalar_ops": category_count(("s_",)),
        "buffer_ops": category_count(("buffer_", "tbuffer_")),
        "lds_ops": category_count(("ds_",)),
        "flat_ops": category_count(("flat_", "global_")),
        "image_ops": category_count(("image_",)),
        "wait_ops": counts.get("s_waitcnt", 0),
        "branch_ops": sum(count for opcode, count in counts.items() if opcode.startswith(("s_branch", "s_cbranch"))),
    }


def generate_advice(report: dict[str, Any]) -> list[str]:
    advice: list[str] = []
    counts = chunk_counts(report)
    events = flattened_events(report)
    code_objects = code_object_summary(report)
    pso = pso_summary(report)
    sqtt = sqtt_summary(report)

    if not report["header"]["valid_magic"]:
        advice.append("The file magic is invalid; this file does not look like a valid .rgp capture.")
        return advice

    if not events:
        advice.append("No queue events were decoded.")
    else:
        slowest = max(events, key=lambda item: item["gpu_duration"])
        advice.append(
            "Slowest submit: "
            f"{slowest['event_name']} on {slowest['queue_type_name']}/{slowest['engine_type_name']} "
            f"with gpu_duration={slowest['gpu_duration']}."
        )

        compute_events = [event for event in events if event["queue_type_name"] == "COMPUTE"]
        if compute_events:
            advice.append(f"The capture includes {len(compute_events)} compute-queue events.")
        else:
            advice.append("No compute queue events were identified.")

    if "SQTT_DATA" not in counts:
        advice.append("No SQTT_DATA chunks were found.")
    else:
        advice.append(f"The capture contains {sqtt['data_chunk_count']} SQTT streams with {sqtt['total_trace_bytes']} bytes of trace payload.")

    if "SPM_DB" not in counts:
        advice.append("No SPM_DB chunk was found.")

    if code_objects["record_count"] == 0:
        advice.append("No code object records were parsed.")
    else:
        advice.append(f"Parsed {code_objects['record_count']} code object records, including {code_objects['elf_record_count']} ELF payloads.")

    if pso["record_count"] == 0:
        advice.append("No PSO correlation records were decoded.")
    else:
        advice.append(f"Decoded {pso['record_count']} PSO correlation records.")

    corr = correlation_summary(report)
    if corr["pso_record_count"] and corr["matched_pso_count"]:
        advice.append(f"Matched {corr['matched_pso_count']} PSO records to loader-event hashes.")
    elif corr["pso_record_count"]:
        advice.append("PSO records exist, but none matched loader-event hashes.")

    if len(events) == 1:
        advice.append("Only one queue event was recorded.")
    return advice
