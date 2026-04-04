from __future__ import annotations

from typing import Any

from .tinygrad_support.isa_map import _instructions_for_code_object


def load_disassembly(
    report: dict[str, Any] | None,
    focus_index: int | None,
    isa_tool: str | None,
) -> dict[int, dict[str, Any]]:
    if report is None or focus_index is None:
        return {}
    try:
        instructions = _instructions_for_code_object(report, code_object_index=int(focus_index), tool=isa_tool)
    except Exception:
        return {}
    return {
        int(item.address): {
            "text": item.text,
            "mnemonic": item.mnemonic,
            "operands": item.operands,
            "size": item.size,
            "branch_target": item.branch_target,
        }
        for item in instructions
    }


def annotate_pc(item: dict[str, Any], disassembly: dict[int, dict[str, Any]]) -> dict[str, Any]:
    annotated = dict(item)
    pc = int(item.get("pc", 0) or 0)
    static = disassembly.get(pc)
    if static:
        annotated["text"] = static.get("text")
        annotated["size"] = static.get("size")
        annotated["branch_target"] = static.get("branch_target")
    return annotated


def focus_hotspot_candidates(
    decode_stream: dict[str, Any],
    focus_index: int | None,
    disassembly: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if focus_index is None:
        return []
    rows: list[dict[str, Any]] = []
    for hotspot in decode_stream.get("annotated_hotspots") or []:
        for candidate in hotspot.get("stitched_candidates") or []:
            if int(candidate.get("code_object_index", -1) or -1) != int(focus_index):
                continue
            symbol = candidate.get("symbol") or {}
            rows.append(
                {
                    "address": hotspot.get("address"),
                    "hitcount": hotspot.get("hitcount"),
                    "total_duration": hotspot.get("total_duration"),
                    "total_stall": hotspot.get("total_stall"),
                    "avg_duration_per_hit": (hotspot.get("total_duration", 0) or 0)
                    / max(int(hotspot.get("hitcount", 0) or 0), 1),
                    "avg_stall_per_hit": (hotspot.get("total_stall", 0) or 0)
                    / max(int(hotspot.get("hitcount", 0) or 0), 1),
                    "dispatch_assignment_share": candidate.get("dispatch_assignment_share"),
                    "dispatch_isa_mapped_dispatch_share": candidate.get("dispatch_isa_mapped_dispatch_share"),
                    "match_kind": candidate.get("match_kind"),
                    "bucket": "kernel_entry" if int(hotspot.get("address", 0) or 0) == 0 else None,
                    "symbol": {
                        "name": symbol.get("name"),
                        "offset": symbol.get("offset"),
                    },
                    "top_pcs": [
                        annotate_pc(
                            {
                                "code_object_index": int(focus_index),
                                "pc": item.get("pc"),
                                "mnemonic": item.get("mnemonic"),
                                "operands": item.get("operands"),
                                "category": item.get("category"),
                                "count": item.get("count"),
                            },
                            disassembly,
                        )
                        for item in (candidate.get("dispatch_isa_top_pcs") or [])[:4]
                    ],
                }
            )
    rows.sort(
        key=lambda item: (
            -int(item.get("total_duration", 0) or 0),
            -float(item.get("dispatch_assignment_share", 0.0) or 0.0),
            -int(item.get("hitcount", 0) or 0),
        )
    )
    return rows


def build_instruction_ranking(
    top_pcs: list[dict[str, Any]],
    hotspot_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranking: dict[str, dict[str, Any]] = {}

    def add_pc(item: dict[str, Any], *, hotspot_weight: int = 0) -> None:
        pc = int(item.get("pc", 0) or 0)
        signature = f"0x{pc:x}"
        row = ranking.setdefault(
            signature,
            {
                "pc": pc,
                "mnemonic": item.get("mnemonic"),
                "operands": item.get("operands"),
                "category": item.get("category"),
                "text": item.get("text"),
                "dispatch_count": 0,
                "hotspot_mentions": 0,
                "hotspot_duration": 0,
                "hotspot_stall": 0,
                "score": 0,
            },
        )
        row["dispatch_count"] += int(item.get("count", 0) or 0)
        if hotspot_weight > 0:
            row["hotspot_mentions"] += 1
            row["score"] += hotspot_weight
        runtime_duration = int(item.get("runtime_duration", 0) or 0)
        runtime_stall = int(item.get("runtime_stall", 0) or 0)
        row["hotspot_duration"] += runtime_duration
        row["hotspot_stall"] += runtime_stall
        row["score"] += runtime_duration + runtime_stall
        row["score"] += int(item.get("count", 0) or 0)
        if not row.get("text") and item.get("text"):
            row["text"] = item.get("text")

    for item in top_pcs:
        add_pc(item)
    for hotspot in hotspot_candidates:
        weight = max(1, int(hotspot.get("hitcount", 0) or 0))
        for item in hotspot.get("top_pcs") or []:
            enriched = dict(item)
            enriched["runtime_duration"] = hotspot.get("total_duration")
            enriched["runtime_stall"] = hotspot.get("total_stall")
            add_pc(enriched, hotspot_weight=weight)

    rows = list(ranking.values())
    rows.sort(key=lambda item: (-int(item.get("score", 0) or 0), -int(item.get("dispatch_count", 0) or 0), int(item.get("pc", 0) or 0)))
    return rows[:8]


def numeric_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    delta = after - before
    payload: dict[str, Any] = {"before": before, "after": after, "delta": delta}
    if before not in (0, 0.0):
        payload["delta_ratio"] = delta / before
    return payload


def pc_ranking_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_pc_signature(item): item for item in before_items}
    after_map = {_pc_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = int((before_map.get(signature) or {}).get("count", 0) or 0)
        after = int((after_map.get(signature) or {}).get("count", 0) or 0)
        rows.append({"signature": signature, "before": before, "after": after, "delta": after - before})
    rows.sort(key=lambda item: (-abs(item["delta"]), -item["after"], item["signature"]))
    return rows[:6]


def instruction_ranking_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_instruction_signature(item): item for item in before_items}
    after_map = {_instruction_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = before_map.get(signature) or {}
        after = after_map.get(signature) or {}
        rows.append(
            {
                "signature": signature,
                "score": numeric_delta(before.get("score"), after.get("score")),
                "dispatch_count": numeric_delta(before.get("dispatch_count"), after.get("dispatch_count")),
                "hotspot_mentions": numeric_delta(before.get("hotspot_mentions"), after.get("hotspot_mentions")),
                "hotspot_duration": numeric_delta(before.get("hotspot_duration"), after.get("hotspot_duration")),
                "hotspot_stall": numeric_delta(before.get("hotspot_stall"), after.get("hotspot_stall")),
                "category": after.get("category") or before.get("category"),
                "text": after.get("text") or before.get("text"),
            }
        )
    rows.sort(
        key=lambda item: (
            -abs(((item.get("score") or {}).get("delta") or 0)),
            -abs(((item.get("hotspot_duration") or {}).get("delta") or 0)),
            -abs(((item.get("dispatch_count") or {}).get("delta") or 0)),
            item["signature"],
        )
    )
    return rows[:8]


def runtime_hotspot_ranking_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_runtime_hotspot_signature(item): item for item in before_items}
    after_map = {_runtime_hotspot_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = int((before_map.get(signature) or {}).get("total_duration", 0) or 0)
        after = int((after_map.get(signature) or {}).get("total_duration", 0) or 0)
        rows.append({"signature": signature, "before": before, "after": after, "delta": after - before})
    rows.sort(key=lambda item: (-abs(item["delta"]), -item["after"], item["signature"]))
    return rows[:6]


def focused_runtime_hotspot_deltas(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {_focused_runtime_hotspot_signature(item): item for item in before_items}
    after_map = {_focused_runtime_hotspot_signature(item): item for item in after_items}
    rows: list[dict[str, Any]] = []
    for signature in sorted(set(before_map) | set(after_map)):
        before = before_map.get(signature) or {}
        after = after_map.get(signature) or {}
        row: dict[str, Any] = {
            "signature": signature,
            "duration": numeric_delta(before.get("total_duration"), after.get("total_duration")),
            "stall": numeric_delta(before.get("total_stall"), after.get("total_stall")),
            "hitcount": numeric_delta(before.get("hitcount"), after.get("hitcount")),
            "avg_duration_per_hit": numeric_delta(before.get("avg_duration_per_hit"), after.get("avg_duration_per_hit")),
            "avg_stall_per_hit": numeric_delta(before.get("avg_stall_per_hit"), after.get("avg_stall_per_hit")),
            "dispatch_assignment_share": numeric_delta(before.get("dispatch_assignment_share"), after.get("dispatch_assignment_share")),
            "dispatch_isa_mapped_dispatch_share": numeric_delta(
                before.get("dispatch_isa_mapped_dispatch_share"),
                after.get("dispatch_isa_mapped_dispatch_share"),
            ),
            "top_pcs": pc_ranking_delta(before.get("top_pcs") or [], after.get("top_pcs") or []),
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -abs(((item.get("duration") or {}).get("delta") or 0)),
            -abs(((item.get("stall") or {}).get("delta") or 0)),
            item["signature"],
        )
    )
    return rows[:6]


def _pc_signature(item: dict[str, Any]) -> str:
    return (
        f"code_object[{int(item.get('code_object_index', 0) or 0)}]"
        f":0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic') or ''} {item.get('operands') or ''}".strip()
    )


def _instruction_signature(item: dict[str, Any]) -> str:
    return (
        f"0x{int(item.get('pc', 0) or 0):x} "
        f"{item.get('mnemonic') or ''} {item.get('operands') or ''}"
    ).strip()


def _runtime_hotspot_signature(item: dict[str, Any]) -> str:
    symbol = item.get("symbol") or {}
    if symbol.get("name"):
        return f"{symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
    return f"0x{int(item.get('address', 0) or 0):x}"


def _focused_runtime_hotspot_signature(item: dict[str, Any]) -> str:
    symbol = item.get("symbol") or {}
    bucket = item.get("bucket")
    if symbol.get("name"):
        signature = f"{symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
    else:
        signature = f"0x{int(item.get('address', 0) or 0):x}"
    if bucket:
        signature += f" [{bucket}]"
    return signature
