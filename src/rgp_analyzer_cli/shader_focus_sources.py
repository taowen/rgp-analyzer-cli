from __future__ import annotations

from pathlib import Path
from typing import Any


def build_source_hints(
    source_file: Path | None,
    runtime_profile: dict[str, Any],
    hotspot_candidates: list[dict[str, Any]],
    top_pcs: list[dict[str, Any]],
) -> dict[str, Any]:
    if source_file is None:
        return {}
    path = Path(source_file)
    if not path.exists() or not path.is_file():
        return {"file": str(path), "available": False}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"file": str(path), "available": False}

    reasons: list[tuple[str, tuple[str, ...]]] = []
    sync_wait_share = runtime_profile.get("sync_wait_share")
    immed_stall_per_inst = runtime_profile.get("immed_stall_per_inst")
    lds_stall_per_inst = runtime_profile.get("lds_stall_per_inst")
    global_mem_share = runtime_profile.get("global_memory_duration_share")

    if isinstance(sync_wait_share, (int, float)) and sync_wait_share >= 0.2:
        reasons.append(("sync_wait", ("barrier(", "subgroupShuffleXor", "row_split", "SubGroupSize", "tmpsh", "tmpshv4")))
    if isinstance(immed_stall_per_inst, (int, float)) and immed_stall_per_inst >= 10:
        reasons.append(("immed_sync", ("barrier(", "subgroupShuffleXor", "row_split", "SubGroupSize")))
    if isinstance(lds_stall_per_inst, (int, float)) and lds_stall_per_inst > 0:
        reasons.append(("lds_pressure", ("shared ", "tmpsh", "tmpshv4", "kvsh", "barrier(")))
    if isinstance(global_mem_share, (int, float)) and global_mem_share >= 0.05:
        reasons.append(("global_memory", ("dequantize", "data_q", "data_k", "data_v", "kvsh", "SHMEM_STAGING")))
    if not reasons:
        reasons.append(("control_flow", ("barrier(", "subgroupShuffleXor", "if (", "for (", "while (")))

    matches: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for reason, needles in reasons:
        for lineno, line in enumerate(lines, start=1):
            if lineno in seen_lines:
                continue
            if not line.strip():
                continue
            if any(needle in line for needle in needles):
                seen_lines.add(lineno)
                start = max(1, lineno - 1)
                end = min(len(lines), lineno + 1)
                excerpt = [
                    {"line": idx, "text": lines[idx - 1].rstrip(), "focus": idx == lineno}
                    for idx in range(start, end + 1)
                ]
                matches.append(
                    {
                        "reason": reason,
                        "line": lineno,
                        "match": line.strip(),
                        "excerpt": excerpt,
                    }
                )
            if len(matches) >= 10:
                break
        if len(matches) >= 10:
            break

    pc_notes = []
    for item in top_pcs[:4]:
        note = f"0x{int(item.get('pc', 0) or 0):x} {item.get('mnemonic')} {item.get('operands')}".strip()
        pc_notes.append(note)
    for item in hotspot_candidates[:2]:
        if int(item.get("address", 0) or 0) == 0:
            pc_notes.append("kernel_entry_hotspot_bucket")
            break

    return {
        "file": str(path),
        "available": True,
        "match_count": len(matches),
        "matches": matches,
        "pc_notes": pc_notes,
    }


def build_source_delta_hints(
    baseline_source_hints: dict[str, Any],
    candidate_source_hints: dict[str, Any],
    runtime_proxy_deltas: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_matches = candidate_source_hints.get("matches") or []
    baseline_matches = baseline_source_hints.get("matches") or []
    source_matches = candidate_matches or baseline_matches
    if not source_matches:
        return []

    prioritized_reasons: list[str] = []
    sync_delta = (runtime_proxy_deltas.get("sync_wait_cycles_per_inst") or {}).get("delta")
    immed_delta = (runtime_proxy_deltas.get("immed_stall_per_inst") or {}).get("delta")
    global_mem_delta = (runtime_proxy_deltas.get("global_memory_duration_share") or {}).get("delta")
    lds_delta = (runtime_proxy_deltas.get("lds_stall_per_inst") or {}).get("delta")

    if isinstance(sync_delta, (int, float)) and abs(sync_delta) >= 0.1:
        prioritized_reasons.extend(["sync_wait", "immed_sync"])
    if isinstance(immed_delta, (int, float)) and abs(immed_delta) >= 1.0:
        prioritized_reasons.extend(["immed_sync", "sync_wait"])
    if isinstance(global_mem_delta, (int, float)) and abs(global_mem_delta) >= 0.01:
        prioritized_reasons.append("global_memory")
    if isinstance(lds_delta, (int, float)) and abs(lds_delta) >= 0.1:
        prioritized_reasons.append("lds_pressure")

    if not prioritized_reasons:
        prioritized_reasons.extend(["sync_wait", "immed_sync", "global_memory", "lds_pressure", "control_flow"])

    rows: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for reason in prioritized_reasons:
        for item in source_matches:
            if item.get("reason") != reason:
                continue
            line = int(item.get("line", 0) or 0)
            if line in seen_lines:
                continue
            seen_lines.add(line)
            rows.append(
                {
                    "reason": reason,
                    "line": line,
                    "match": item.get("match"),
                    "excerpt": item.get("excerpt") or [],
                }
            )
            if len(rows) >= 6:
                return rows
    return rows
