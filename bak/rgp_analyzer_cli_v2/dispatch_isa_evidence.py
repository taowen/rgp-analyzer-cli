from __future__ import annotations

from typing import Any

from .tinygrad_support.isa_map import map_dispatch_spans_to_isa


def build_stream_dispatch_isa_summary(report: dict[str, Any], stream_index: int) -> dict[str, Any] | None:
    try:
        result = map_dispatch_spans_to_isa(
            report,
            stream_index=stream_index,
            dispatch_limit=0,
            context_packets=64,
            packet_limit=5000,
            mapped_limit=24,
        )
    except Exception as exc:
        return {
            "stream_index": stream_index,
            "error": str(exc),
            "dispatch_count": 0,
            "mapped_dispatch_count": 0,
            "code_objects": {},
        }

    code_objects: dict[int, dict[str, Any]] = {}
    mapped_dispatch_count = 0
    for dispatch in result.get("dispatches", []):
        code_object_index = dispatch.get("code_object_index")
        if not isinstance(code_object_index, int):
            continue
        entry = code_objects.setdefault(
            code_object_index,
            {
                "dispatch_count": 0,
                "mapped_dispatch_count": 0,
                "mapped_packet_count": 0,
                "execution_packet_count": 0,
                "top_pcs": [],
            },
        )
        entry["dispatch_count"] += 1
        entry["execution_packet_count"] += int(dispatch.get("execution_packet_count", 0) or 0)
        if int(dispatch.get("mapped_count", 0) or 0) > 0:
            mapped_dispatch_count += 1
            entry["mapped_dispatch_count"] += 1
            entry["mapped_packet_count"] += int(dispatch.get("mapped_count", 0) or 0)
            for item in dispatch.get("pc_summary", [])[:4]:
                entry["top_pcs"].append(
                    {
                        "pc": int(item["pc"]),
                        "mnemonic": item["mnemonic"],
                        "operands": item["operands"],
                        "category": item.get("category"),
                        "count": int(item["count"]),
                        "dispatch_index": int(dispatch.get("dispatch_index", 0) or 0),
                    }
                )

    for entry in code_objects.values():
        entry["top_pcs"].sort(
            key=lambda item: (-int(item["count"]), int(item["pc"]), int(item["dispatch_index"]))
        )
        entry["top_pcs"] = entry["top_pcs"][:8]
        dispatch_count = int(entry["dispatch_count"])
        mapped_count = int(entry["mapped_dispatch_count"])
        entry["mapped_dispatch_share"] = (mapped_count / dispatch_count) if dispatch_count else None

    return {
        "stream_index": stream_index,
        "dispatch_count": int(result.get("dispatch_count", 0) or 0),
        "mapped_dispatch_count": mapped_dispatch_count,
        "code_objects": code_objects,
    }


def build_dispatch_isa_overview(dispatch_isa_by_stream: dict[int, dict[str, Any] | None]) -> dict[str, Any]:
    code_objects: dict[int, dict[str, Any]] = {}
    total_dispatch_count = 0
    total_mapped_dispatch_count = 0
    for stream_summary in dispatch_isa_by_stream.values():
        if not stream_summary:
            continue
        total_dispatch_count += int(stream_summary.get("dispatch_count", 0) or 0)
        total_mapped_dispatch_count += int(stream_summary.get("mapped_dispatch_count", 0) or 0)
        for code_object_index, entry in ((stream_summary.get("code_objects") or {}).items()):
            target = code_objects.setdefault(
                int(code_object_index),
                {
                    "dispatch_count": 0,
                    "mapped_dispatch_count": 0,
                    "mapped_packet_count": 0,
                    "execution_packet_count": 0,
                    "top_pcs": [],
                },
            )
            target["dispatch_count"] += int(entry.get("dispatch_count", 0) or 0)
            target["mapped_dispatch_count"] += int(entry.get("mapped_dispatch_count", 0) or 0)
            target["mapped_packet_count"] += int(entry.get("mapped_packet_count", 0) or 0)
            target["execution_packet_count"] += int(entry.get("execution_packet_count", 0) or 0)
            target["top_pcs"].extend(entry.get("top_pcs", []))

    for entry in code_objects.values():
        entry["top_pcs"].sort(
            key=lambda item: (-int(item["count"]), int(item["pc"]), int(item["dispatch_index"]))
        )
        entry["top_pcs"] = entry["top_pcs"][:8]
        dispatch_count = int(entry["dispatch_count"])
        mapped_count = int(entry["mapped_dispatch_count"])
        entry["mapped_dispatch_share"] = (mapped_count / dispatch_count) if dispatch_count else None

    ordered = sorted(
        (
            {
                "code_object_index": code_object_index,
                "dispatch_count": int(entry["dispatch_count"]),
                "mapped_dispatch_count": int(entry["mapped_dispatch_count"]),
                "mapped_dispatch_share": entry.get("mapped_dispatch_share"),
                "mapped_packet_count": int(entry["mapped_packet_count"]),
                "top_pc": (entry.get("top_pcs") or [None])[0],
            }
            for code_object_index, entry in code_objects.items()
        ),
        key=lambda item: (
            -int(item["mapped_dispatch_count"]),
            -int(item["mapped_packet_count"]),
            int(item["code_object_index"]),
        ),
    )

    return {
        "dispatch_count": total_dispatch_count,
        "mapped_dispatch_count": total_mapped_dispatch_count,
        "code_objects": code_objects,
        "ordered": ordered,
    }
