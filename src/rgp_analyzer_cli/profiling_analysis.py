from __future__ import annotations

from typing import Any


def build_runtime_profile(stream: dict[str, Any]) -> dict[str, Any]:
    instructions = int(stream.get("instructions", 0) or 0)
    total_duration = int(stream.get("total_instruction_duration", 0) or 0)
    total_stall = int(stream.get("total_instruction_stall", 0) or 0)
    instructions_with_stall = int(stream.get("instructions_with_stall", 0) or 0)
    wave_records = int(stream.get("wave_records", 0) or 0)
    total_wave_lifetime = int(stream.get("total_wave_lifetime", 0) or 0)
    occupancy_begin = int(stream.get("occupancy_begin_time", 0) or 0)
    occupancy_end = int(stream.get("occupancy_end_time", 0) or 0)
    occupancy_span = max(0, occupancy_end - occupancy_begin)
    occupancy_weighted = int(stream.get("occupancy_weighted_time", 0) or 0)
    category_counts = {str(k): int(v) for k, v in (stream.get("category_counts") or {}).items()}
    category_duration_totals = {str(k): int(v) for k, v in (stream.get("category_duration_totals") or {}).items()}
    category_stall_totals = {str(k): int(v) for k, v in (stream.get("category_stall_totals") or {}).items()}
    wave_state_durations = {str(k): int(v) for k, v in (stream.get("wave_state_durations") or {}).items()}

    category_profiles = []
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        duration = category_duration_totals.get(category, 0)
        stall = category_stall_totals.get(category, 0)
        category_profiles.append(
            {
                "category": category,
                "count": count,
                "count_share": (count / instructions) if instructions else None,
                "duration_total": duration,
                "stall_total": stall,
                "duration_share": (duration / total_duration) if total_duration else None,
                "stall_share": (stall / total_stall) if total_stall else None,
                "duration_per_inst": (duration / count) if count else None,
                "stall_per_inst": (stall / count) if count else None,
                "stall_share_of_duration": (stall / duration) if duration else None,
            }
        )

    wave_state_profiles = []
    total_state_duration = sum(wave_state_durations.values())
    for state, duration in sorted(wave_state_durations.items(), key=lambda item: (-item[1], item[0])):
        wave_state_profiles.append(
            {
                "state": state,
                "duration": duration,
                "share": (duration / total_state_duration) if total_state_duration else None,
            }
        )

    occupancy_average_active = (occupancy_weighted / occupancy_span) if occupancy_span > 0 else None
    memory_categories = ("VMEM", "SMEM", "LDS")
    global_memory_categories = ("VMEM", "SMEM")
    lds_categories = ("LDS",)
    immed_categories = ("IMMED",)
    sync_wait_states = ("WAIT",)
    memory_instruction_count = sum(category_counts.get(name, 0) for name in memory_categories)
    memory_duration_total = sum(category_duration_totals.get(name, 0) for name in memory_categories)
    memory_stall_total = sum(category_stall_totals.get(name, 0) for name in memory_categories)
    global_memory_instruction_count = sum(category_counts.get(name, 0) for name in global_memory_categories)
    global_memory_duration_total = sum(category_duration_totals.get(name, 0) for name in global_memory_categories)
    global_memory_stall_total = sum(category_stall_totals.get(name, 0) for name in global_memory_categories)
    lds_instruction_count = sum(category_counts.get(name, 0) for name in lds_categories)
    lds_duration_total = sum(category_duration_totals.get(name, 0) for name in lds_categories)
    lds_stall_total = sum(category_stall_totals.get(name, 0) for name in lds_categories)
    immed_instruction_count = sum(category_counts.get(name, 0) for name in immed_categories)
    immed_duration_total = sum(category_duration_totals.get(name, 0) for name in immed_categories)
    immed_stall_total = sum(category_stall_totals.get(name, 0) for name in immed_categories)
    scalar_duration_total = (
        category_duration_totals.get("SALU", 0)
        + category_duration_totals.get("SMEM", 0)
        + category_duration_totals.get("MESSAGE", 0)
    )
    scalar_stall_total = (
        category_stall_totals.get("SALU", 0)
        + category_stall_totals.get("SMEM", 0)
        + category_stall_totals.get("MESSAGE", 0)
    )
    vector_duration_total = (
        category_duration_totals.get("VALU", 0)
        + category_duration_totals.get("VMEM", 0)
        + category_duration_totals.get("LDS", 0)
    )
    vector_stall_total = (
        category_stall_totals.get("VALU", 0)
        + category_stall_totals.get("VMEM", 0)
        + category_stall_totals.get("LDS", 0)
    )
    sync_wait_duration_total = sum(wave_state_durations.get(name, 0) for name in sync_wait_states)
    sync_wait_per_inst = (sync_wait_duration_total / instructions) if instructions else None
    immed_stall_per_inst = (immed_stall_total / immed_instruction_count) if immed_instruction_count else None
    lds_stall_per_inst = (lds_stall_total / lds_instruction_count) if lds_instruction_count else None
    return {
        "instructions": instructions,
        "instructions_with_stall": instructions_with_stall,
        "total_instruction_duration": total_duration,
        "total_instruction_stall": total_stall,
        "avg_duration_per_inst": (total_duration / instructions) if instructions else None,
        "avg_stall_per_inst": (total_stall / instructions) if instructions else None,
        "stalled_instruction_share": (instructions_with_stall / instructions) if instructions else None,
        "stall_share_of_duration": (total_stall / total_duration) if total_duration else None,
        "wave_records": wave_records,
        "avg_wave_lifetime": (total_wave_lifetime / wave_records) if wave_records else None,
        "max_wave_lifetime": int(stream.get("max_wave_lifetime", 0) or 0),
        "occupancy_max_active": int(stream.get("occupancy_max_active", 0) or 0),
        "occupancy_average_active": occupancy_average_active,
        "occupancy_span": occupancy_span,
        "memory_instruction_share": (memory_instruction_count / instructions) if instructions else None,
        "memory_duration_share": (memory_duration_total / total_duration) if total_duration else None,
        "memory_stall_share": (memory_stall_total / total_stall) if total_stall else None,
        "global_memory_instruction_share": (global_memory_instruction_count / instructions) if instructions else None,
        "global_memory_duration_share": (global_memory_duration_total / total_duration) if total_duration else None,
        "global_memory_stall_share": (global_memory_stall_total / total_stall) if total_stall else None,
        "lds_instruction_share": (lds_instruction_count / instructions) if instructions else None,
        "lds_duration_share": (lds_duration_total / total_duration) if total_duration else None,
        "lds_stall_share": (lds_stall_total / total_stall) if total_stall else None,
        "scalar_duration_share": (scalar_duration_total / total_duration) if total_duration else None,
        "scalar_stall_share": (scalar_stall_total / total_stall) if total_stall else None,
        "vector_duration_share": (vector_duration_total / total_duration) if total_duration else None,
        "vector_stall_share": (vector_stall_total / total_stall) if total_stall else None,
        "sync_wait_share": (sync_wait_duration_total / total_state_duration) if total_state_duration else None,
        "sync_wait_cycles": sync_wait_duration_total,
        "sync_wait_cycles_per_inst": sync_wait_per_inst,
        "immed_instruction_share": (immed_instruction_count / instructions) if instructions else None,
        "immed_duration_share": (immed_duration_total / total_duration) if total_duration else None,
        "immed_stall_share": (immed_stall_total / total_stall) if total_stall else None,
        "immed_stall_per_inst": immed_stall_per_inst,
        "lds_stall_per_inst": lds_stall_per_inst,
        "category_profiles": category_profiles,
        "wave_state_profiles": wave_state_profiles,
    }


def build_runtime_hotspot_profiles(annotated_hotspots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for hotspot in annotated_hotspots:
        summary = hotspot.get("stitch_summary") or {}
        candidates = hotspot.get("stitched_candidates") or []
        primary = candidates[0] if candidates else {}
        symbol = primary.get("symbol") or {}
        hitcount = int(hotspot.get("hitcount", 0) or 0)
        total_duration = int(hotspot.get("total_duration", 0) or 0)
        total_stall = int(hotspot.get("total_stall", 0) or 0)
        profiles.append(
            {
                "code_object_index": summary.get("primary_code_object_index"),
                "match_kind": summary.get("primary_match_kind"),
                "address": int(hotspot.get("address", 0) or 0),
                "hitcount": hitcount,
                "total_duration": total_duration,
                "total_stall": total_stall,
                "avg_duration_per_hit": (total_duration / hitcount) if hitcount else None,
                "avg_stall_per_hit": (total_stall / hitcount) if hitcount else None,
                "stall_share_of_duration": (total_stall / total_duration) if total_duration else None,
                "symbol": {
                    "name": symbol.get("name") or summary.get("primary_symbol"),
                    "offset": symbol.get("offset", summary.get("primary_symbol_offset")),
                }
                if (symbol.get("name") or summary.get("primary_symbol"))
                else None,
            }
        )
    profiles.sort(
        key=lambda item: (
            -int(item.get("total_duration", 0) or 0),
            -int(item.get("total_stall", 0) or 0),
            -int(item.get("hitcount", 0) or 0),
            int(item.get("address", 0) or 0),
        )
    )
    return profiles[:8]
