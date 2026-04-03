from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..decode_bridge import collect_code_object_records
from ..parser import materialize_code_object_payload
from ..stitch_model import build_stitch_model
from .common import resolve_tinygrad_path, tinygrad_import_path


@dataclass(frozen=True)
class StaticInstruction:
    address: int
    size: int
    mnemonic: str
    operands: str
    text: str
    branch_target: int | None = None


_DISASM_LINE_RE = re.compile(r"^\s*([a-z0-9_\.]+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*(.*)$")
_LABEL_TARGET_RE = re.compile(r"<[^>]+\+0x([0-9A-Fa-f]+)>")


def _safe_packet_op_name(packet: Any) -> str:
    try:
        op = getattr(packet, "op", None)
    except Exception:
        return ""
    return getattr(op, "name", "")


def _mnemonic_category(mnemonic: str) -> str:
    lower = mnemonic.lower()
    if lower.startswith(("s_cbranch", "s_branch")):
        return "branch"
    if lower.startswith(("buffer_", "global_", "flat_", "scratch_", "smem_", "s_load", "s_store")):
        return "memory"
    if lower.startswith("ds_"):
        return "lds"
    if lower.startswith("s_"):
        return "scalar"
    if lower.startswith("v_"):
        return "vector"
    return "other"


def _parse_disassembly(text: str) -> list[StaticInstruction]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _DISASM_LINE_RE.match(line)
        if not match:
            continue
        mnemonic = match.group(1)
        operands = match.group(2).strip()
        address = int(match.group(3), 16)
        branch_target = None
        if "<" in operands and ">" in operands:
            label_match = _LABEL_TARGET_RE.search(operands)
            if label_match is not None:
                branch_target = int(label_match.group(1), 16)
        rows.append(
            {
                "address": address,
                "mnemonic": mnemonic,
                "operands": operands,
                "text": line.strip(),
                "branch_target": branch_target,
            }
        )
    instructions: list[StaticInstruction] = []
    for index, row in enumerate(rows):
        next_address = rows[index + 1]["address"] if index + 1 < len(rows) else row["address"] + 4
        size = max(4, next_address - row["address"])
        instructions.append(
            StaticInstruction(
                address=row["address"],
                size=size,
                mnemonic=row["mnemonic"],
                operands=row["operands"],
                text=row["text"],
                branch_target=row["branch_target"],
            )
        )
    return instructions


def _disassemble_payload(payload: bytes, *, tool: str | None = None, symbol: str = "_amdgpu_cs_main") -> list[StaticInstruction]:
    objdump = shutil.which(tool) if tool else None
    objdump = objdump or shutil.which("llvm-objdump-18") or shutil.which("llvm-objdump")
    if objdump is None:
        raise RuntimeError("llvm-objdump tool not found")
    with tempfile.TemporaryDirectory(prefix="rgp-analyzer-tinygrad-map-") as tmpdir:
        path = Path(tmpdir) / "code-object.elf"
        path.write_bytes(payload)
        proc = subprocess.run(
            [objdump, "-d", f"--disassemble-symbols={symbol}", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    instructions = _parse_disassembly(proc.stdout)
    if not instructions:
        raise RuntimeError("failed to parse llvm-objdump disassembly")
    return instructions


def _instructions_for_code_object(
    report: dict[str, Any],
    *,
    code_object_index: int,
    tool: str | None = None,
    symbol: str = "_amdgpu_cs_main",
) -> list[StaticInstruction]:
    records = collect_code_object_records(report)
    record = next((item for item in records if int(item["index"]) == int(code_object_index)), None)
    if record is None:
        raise RuntimeError(f"code object {code_object_index} not found")
    payload = materialize_code_object_payload(report, record, pad_elf=True)
    return _disassemble_payload(payload, tool=tool, symbol=symbol)


def _map_blob_packets_to_isa(
    blob: bytes,
    instructions: list[StaticInstruction],
    *,
    tinygrad_path: Path | None,
    packet_start: int | None = None,
    packet_end: int | None = None,
    packet_limit: int = 20000,
    mapped_limit: int = 64,
) -> dict[str, Any]:
    resolved = resolve_tinygrad_path(tinygrad_path)
    inst_map = {inst.address: inst for inst in instructions}
    base_pc = instructions[0].address
    with tinygrad_import_path(resolved):
        from tinygrad.renderer.amd.sqtt import decode

        wave_pc: dict[int, int] = {}
        mapped: list[dict[str, Any]] = []
        pc_counts: Counter[tuple[int, str, str]] = Counter()
        category_counts: Counter[str] = Counter()
        packet_type_counts: Counter[str] = Counter()
        packet_count = 0
        considered_packets = 0
        inferred_wave_start_count = 0
        for packet_index, packet in enumerate(decode(blob)):
            packet_count += 1
            if packet_start is not None and packet_index < packet_start:
                continue
            if packet_end is not None and packet_index > packet_end:
                break
            considered_packets += 1
            packet_type = type(packet).__name__
            packet_type_counts[packet_type] += 1
            if packet_type in {"WAVESTART", "WAVESTART_RDNA4", "CDNA_WAVESTART"}:
                wave_pc[getattr(packet, "wave")] = base_pc
            elif packet_type in {"WAVEEND", "CDNA_WAVEEND"}:
                wave_pc.pop(getattr(packet, "wave"), None)
            elif packet_type == "IMMEDIATE_MASK":
                mask = getattr(packet, "mask", 0)
                for wave in range(16):
                    if not (mask & (1 << wave)) or wave not in wave_pc:
                        continue
                    current = inst_map.get(wave_pc[wave])
                    if current is None:
                        continue
                    mapped.append(
                        {
                            "packet_index": packet_index,
                            "packet_type": packet_type,
                            "time": getattr(packet, "_time", None),
                            "wave": wave,
                            "pc": current.address,
                            "mnemonic": current.mnemonic,
                            "operands": current.operands,
                            "text": current.text,
                        }
                    )
                    pc_counts[(current.address, current.mnemonic, current.operands)] += 1
                    category_counts[_mnemonic_category(current.mnemonic)] += 1
                    wave_pc[wave] = current.branch_target if current.branch_target is not None else current.address + current.size
                    if len(mapped) >= mapped_limit:
                        break
            elif packet_type in {"VALUINST", "IMMEDIATE", "INST", "INST_RDNA4"}:
                wave = getattr(packet, "wave", None)
                if wave is None:
                    continue
                if wave not in wave_pc:
                    # Dispatch-local windows often start after the true WAVESTART packet.
                    # Seed the wave at the entry point so we can still recover a conservative
                    # packet-to-ISA view inside the dispatch span.
                    wave_pc[wave] = base_pc
                    inferred_wave_start_count += 1
                current = inst_map.get(wave_pc[wave])
                while current is not None and current.mnemonic in {"s_delay_alu", "s_wait_alu", "s_barrier_wait"}:
                    wave_pc[wave] = current.address + current.size
                    current = inst_map.get(wave_pc[wave])
                if current is None:
                    continue
                if packet_type in {"INST", "INST_RDNA4"} and _safe_packet_op_name(packet).startswith("OTHER_"):
                    continue
                mapped.append(
                    {
                        "packet_index": packet_index,
                        "packet_type": packet_type,
                        "time": getattr(packet, "_time", None),
                        "wave": wave,
                        "pc": current.address,
                        "mnemonic": current.mnemonic,
                        "operands": current.operands,
                        "text": current.text,
                    }
                )
                pc_counts[(current.address, current.mnemonic, current.operands)] += 1
                category_counts[_mnemonic_category(current.mnemonic)] += 1
                packet_op = _safe_packet_op_name(packet)
                if packet_op.startswith("JUMP") and current.branch_target is not None:
                    wave_pc[wave] = current.branch_target
                else:
                    wave_pc[wave] = current.address + current.size
                if len(mapped) >= mapped_limit:
                    break
            if packet_limit > 0 and considered_packets >= packet_limit:
                break
            if len(mapped) >= mapped_limit:
                break
    return {
        "packet_count": packet_count,
        "considered_packets": considered_packets,
        "mapped_count": len(mapped),
        "inferred_wave_start_count": inferred_wave_start_count,
        "packet_type_counts": dict(packet_type_counts),
        "execution_packet_count": sum(
            packet_type_counts.get(name, 0)
            for name in ("VALUINST", "IMMEDIATE", "INST", "INST_RDNA4", "ALUEXEC", "VMEMEXEC", "IMMEDIATE_MASK")
        ),
        "instruction_count": len(instructions),
        "mapped_packets": mapped,
        "pc_summary": [
            {
                "pc": pc,
                "mnemonic": mnemonic,
                "operands": operands,
                "category": _mnemonic_category(mnemonic),
                "count": count,
            }
            for (pc, mnemonic, operands), count in pc_counts.most_common()
        ],
        "category_summary": [
            {"category": category, "count": count}
            for category, count in category_counts.most_common()
        ],
    }


def map_tinygrad_packets_to_isa(
    report: dict[str, Any],
    *,
    tinygrad_path: Path | None = None,
    tool: str | None = None,
    stream_index: int = 0,
    packet_limit: int = 20000,
    mapped_limit: int = 64,
) -> dict[str, Any]:
    records = collect_code_object_records(report)
    if not records:
        raise RuntimeError("no code object records found")
    instructions = _instructions_for_code_object(report, code_object_index=int(records[0]["index"]), tool=tool)
    chunk = report["sqtt_data_chunks"][stream_index]
    blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]
    resolved = resolve_tinygrad_path(tinygrad_path)
    mapped_result = _map_blob_packets_to_isa(
        blob,
        instructions,
        tinygrad_path=resolved,
        packet_limit=packet_limit,
        mapped_limit=mapped_limit,
    )
    return {
        "tinygrad_path": str(resolved),
        "stream_index": stream_index,
        **mapped_result,
    }


def map_dispatch_spans_to_isa(
    report: dict[str, Any],
    *,
    tinygrad_path: Path | None = None,
    tool: str | None = None,
    stream_index: int = 0,
    dispatch_limit: int = 8,
    context_packets: int = 64,
    tail_packets: int = 512,
    packet_limit: int = 20000,
    mapped_limit: int = 32,
) -> dict[str, Any]:
    resolved = resolve_tinygrad_path(tinygrad_path)
    model = build_stitch_model(report)
    stream = next((item for item in model.get("streams", []) if int(item.get("stream_index", -1)) == int(stream_index)), None)
    if stream is None:
        raise RuntimeError(f"stream {stream_index} not found in stitch model")
    spans = stream.get("dispatch_api_spans") or []
    assignments = stream.get("dispatch_assignments") or []
    chunk = report["sqtt_data_chunks"][stream_index]
    blob = report["_blob"][chunk["payload_offset"] : chunk["payload_end"]]

    instructions_cache: dict[int, list[StaticInstruction]] = {}
    dispatches: list[dict[str, Any]] = []
    overall_counts: Counter[tuple[int, str, str, int]] = Counter()

    for dispatch_index, (span, assignment) in enumerate(zip(spans, assignments)):
        if dispatch_limit > 0 and len(dispatches) >= dispatch_limit:
            break
        code_object_index = assignment.get("code_object_index")
        if not isinstance(code_object_index, int):
            continue
        begin_packet = (span.get("begin_marker") or {}).get("packet_index")
        end_packet = (span.get("end_marker") or {}).get("packet_index")
        bind_packet = assignment.get("bind_packet_index")
        if not isinstance(begin_packet, int) or not isinstance(end_packet, int) or end_packet < begin_packet:
            continue
        packet_start = max(0, begin_packet - max(0, context_packets))
        if isinstance(bind_packet, int):
            packet_start = max(bind_packet, packet_start)
        if code_object_index not in instructions_cache:
            instructions_cache[code_object_index] = _instructions_for_code_object(
                report, code_object_index=code_object_index, tool=tool
            )
        packet_end = end_packet + max(0, tail_packets)
        mapped_result = _map_blob_packets_to_isa(
            blob,
            instructions_cache[code_object_index],
            tinygrad_path=resolved,
            packet_start=packet_start,
            packet_end=packet_end,
            packet_limit=packet_limit,
            mapped_limit=mapped_limit,
        )
        for item in mapped_result.get("pc_summary", []):
            overall_counts[(int(item["pc"]), item["mnemonic"], item["operands"], code_object_index)] += int(item["count"])
        unmapped_reason = None
        if int(mapped_result.get("mapped_count", 0)) == 0:
            if int(mapped_result.get("execution_packet_count", 0)) == 0:
                unmapped_reason = "no_execution_packets_in_window"
            elif int(mapped_result.get("inferred_wave_start_count", 0)) == 0:
                unmapped_reason = "no_wave_start_context"
            else:
                unmapped_reason = "execution_packets_not_mapped"
        dispatches.append(
            {
                "dispatch_index": dispatch_index,
                "api_name": assignment.get("api_name"),
                "code_object_index": code_object_index,
                "entry_point": assignment.get("entry_point"),
                "bind_api_pso_hash": assignment.get("bind_api_pso_hash"),
                "bind_packet_index": bind_packet,
                "begin_packet_index": begin_packet,
                "analysis_packet_start": packet_start,
                "analysis_packet_end": packet_end,
                "end_packet_index": end_packet,
                "unmapped_reason": unmapped_reason,
                **mapped_result,
            }
        )

    return {
        "tinygrad_path": str(resolved),
        "stream_index": stream_index,
        "dispatch_count": len(dispatches),
        "context_packets": context_packets,
        "tail_packets": tail_packets,
        "dispatches": dispatches,
        "overall_pc_summary": [
            {
                "pc": pc,
                "mnemonic": mnemonic,
                "operands": operands,
                "code_object_index": code_object_index,
                "category": _mnemonic_category(mnemonic),
                "count": count,
            }
            for (pc, mnemonic, operands, code_object_index), count in overall_counts.most_common()
        ],
    }


def render_tinygrad_isa_map(result: dict[str, Any], limit: int = 32) -> str:
    lines = [
        f"tinygrad_isa_map: path={result['tinygrad_path']} stream={result['stream_index']} "
        f"packets_scanned={result['packet_count']} mapped={result['mapped_count']} static_insts={result['instruction_count']}"
    ]
    for item in result.get("category_summary", [])[: min(limit, 5)]:
        lines.append(f"  category {item['category']} count={item['count']}")
    for item in result.get("pc_summary", [])[: min(limit, 8)]:
        lines.append(
            f"  summary pc=0x{int(item['pc']):x} category={item['category']} count={item['count']} "
            f"{item['mnemonic']} {item['operands']}".rstrip()
        )
    for item in result.get("mapped_packets", [])[:limit]:
        lines.append(
            f"  t={item['time']} wave={item['wave']} packet={item['packet_type']} pc=0x{int(item['pc']):x} "
            f"{item['mnemonic']} {item['operands']}".rstrip()
        )
    return "\n".join(lines)


def render_dispatch_isa_map(result: dict[str, Any], limit: int = 16) -> str:
    lines = [
        f"dispatch_isa_map: path={result['tinygrad_path']} stream={result['stream_index']} dispatches={result['dispatch_count']} "
        f"context_packets={result.get('context_packets')} tail_packets={result.get('tail_packets')}"
    ]
    for item in result.get("overall_pc_summary", [])[: min(limit, 8)]:
        lines.append(
            f"  overall code_object[{item['code_object_index']}] pc=0x{int(item['pc']):x} "
            f"category={item['category']} count={item['count']} {item['mnemonic']} {item['operands']}".rstrip()
        )
    for dispatch in result.get("dispatches", [])[:limit]:
        lines.append(
            f"  dispatch[{dispatch['dispatch_index']}] code_object[{dispatch['code_object_index']}] "
            f"packets={dispatch['analysis_packet_start']}..{dispatch['analysis_packet_end']} "
            f"mapped={dispatch['mapped_count']} inferred_waves={dispatch.get('inferred_wave_start_count', 0)} "
            f"exec_packets={dispatch.get('execution_packet_count', 0)}"
        )
        if dispatch.get("unmapped_reason"):
            lines.append(f"    unmapped_reason={dispatch['unmapped_reason']} packet_types={dispatch.get('packet_type_counts')}")
        for item in dispatch.get("pc_summary", [])[:4]:
            lines.append(
                f"    pc=0x{int(item['pc']):x} category={item['category']} count={item['count']} "
                f"{item['mnemonic']} {item['operands']}".rstrip()
            )
    return "\n".join(lines)
