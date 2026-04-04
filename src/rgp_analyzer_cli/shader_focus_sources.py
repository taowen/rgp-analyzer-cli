from __future__ import annotations

from pathlib import Path
from typing import Any
import re


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


def build_source_isa_blocks(
    source_file: Path | None,
    instructions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if source_file is None:
        return []
    path = Path(source_file)
    if not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    source_patterns: list[tuple[str, tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]]] = [
        (
            "tile_staging",
            (
                re.compile(r"\bktile\b"),
                re.compile(r"\bvtile\b"),
                re.compile(r"\bkscale_tile\b"),
            ),
            (re.compile(r"^ds_(read|write)"), re.compile(r"^buffer_load"), re.compile(r"^global_load"),),
        ),
        (
            "quantized_qk",
            (
                re.compile(r"\bq_i8\b"),
                re.compile(r"\bk_i8\b"),
                re.compile(r"\bq_scale\b"),
                re.compile(r"\bk_scale\b"),
                re.compile(r"\bq_deq\b"),
                re.compile(r"\bkv\b"),
            ),
            (re.compile(r"^v_mul"), re.compile(r"^v_dot"), re.compile(r"^v_mad"), re.compile(r"^v_add"),),
        ),
        (
            "softmax_update",
            (
                re.compile(r"\brow_m\b"),
                re.compile(r"\brow_l\b"),
                re.compile(r"\btile_alpha\b"),
                re.compile(r"\btile_beta\b"),
                re.compile(r"\bexp\s*\("),
                re.compile(r"\bmax\s*\("),
            ),
            (re.compile(r"^v_max"), re.compile(r"^v_exp"), re.compile(r"^s_waitcnt$"), re.compile(r"^v_add"),),
        ),
        (
            "value_accumulate",
            (
                re.compile(r"\bacc\s*="),
                re.compile(r"\bout_data\b"),
                re.compile(r"\bload_v_value\b"),
            ),
            (re.compile(r"^v_mac"), re.compile(r"^v_fmac"), re.compile(r"^v_mul"), re.compile(r"^buffer_store"),),
        ),
        (
            "invocation_index",
            (re.compile(r"\bgl_GlobalInvocationID\.x\b"),),
            (re.compile(r"^v_lshl_add_u32$"), re.compile(r"^v_lshlrev_b32"),),
        ),
        (
            "bounds_check",
            (re.compile(r"\bif\s*\(.*>=.*element_count"), re.compile(r"\breturn\s*;")),
            (re.compile(r"^v_cmp"), re.compile(r"^s_cbranch"),),
        ),
        (
            "push_constant_load",
            (re.compile(r"\bpc\.(element_count|multiplier|bias)\b"),),
            (re.compile(r"^s_load"),),
        ),
        (
            "global_load",
            (re.compile(r"\bin_buf\.data\s*\["), re.compile(r"\binput\w*\s*="),),
            (re.compile(r"^buffer_load"),),
        ),
        (
            "cooperative_matrix",
            (
                re.compile(r"\bcoopmat<"),
                re.compile(r"\bcoopMatLoad\s*\("),
                re.compile(r"\bcoopMatMulAdd\s*\("),
                re.compile(r"\bcoopMatStore\s*\("),
            ),
            (re.compile(r"wmma", re.IGNORECASE), re.compile(r"mfma", re.IGNORECASE),),
        ),
        (
            "value_compute",
            (re.compile(r"\bidx\s*\*\s*pc\.multiplier\b"), re.compile(r"\+\s*pc\.bias\b")),
            (re.compile(r"^v_mul"), re.compile(r"^v_add"),),
        ),
        (
            "shared_exchange",
            (re.compile(r"\bshared\b"), re.compile(r"\bbarrier\s*\("),),
            (re.compile(r"^ds_(read|write)"), re.compile(r"^s_barrier$"), re.compile(r"^s_waitcnt$"),),
        ),
        (
            "buffer_store",
            (re.compile(r"\bout_buf\.data\s*\["),),
            (re.compile(r"^s_waitcnt$"), re.compile(r"^buffer_store"),),
        ),
        (
            "kernel_end",
            (re.compile(r"^\s*}\s*$"),),
            (re.compile(r"^s_sendmsg$"), re.compile(r"^s_endpgm$"),),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for label, source_regexes, isa_regexes in source_patterns:
        matched_lines: list[dict[str, Any]] = []
        for lineno, line in enumerate(lines, start=1):
            if any(regex.search(line) for regex in source_regexes):
                matched_lines.append({"line": lineno, "match": line.strip()})
        if not matched_lines:
            continue

        matched_instructions: list[dict[str, Any]] = []
        for item in instructions:
            mnemonic = str(item.get("mnemonic") or "")
            if any(regex.search(mnemonic) for regex in isa_regexes):
                matched_instructions.append(
                    {
                        "pc": int(item.get("pc", 0) or 0),
                        "mnemonic": mnemonic,
                        "operands": item.get("operands"),
                        "text": item.get("text"),
                    }
                )
        if not matched_instructions:
            continue

        rows.append(
            {
                "label": label,
                "source_lines": matched_lines[:4],
                "isa_instructions": matched_instructions[:6],
            }
        )
    return rows

