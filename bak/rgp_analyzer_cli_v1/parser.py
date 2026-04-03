from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SQTT_FILE_MAGIC_NUMBER = 0x50303042

CHUNK_TYPE_NAMES = {
    0: "ASIC_INFO",
    1: "SQTT_DESC",
    2: "SQTT_DATA",
    3: "API_INFO",
    4: "RESERVED",
    5: "QUEUE_EVENT_TIMINGS",
    6: "CLOCK_CALIBRATION",
    7: "CPU_INFO",
    8: "SPM_DB",
    9: "CODE_OBJECT_DATABASE",
    10: "CODE_OBJECT_LOADER_EVENTS",
    11: "PSO_CORRELATION",
    12: "INSTRUMENTATION_TABLE",
    128: "DERIVED_SPM_DB",
}

QUEUE_EVENT_TYPE_NAMES = {
    0: "CMDBUF_SUBMIT",
    1: "SIGNAL_SEMAPHORE",
    2: "WAIT_SEMAPHORE",
    3: "PRESENT",
}

QUEUE_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "UNIVERSAL",
    2: "COMPUTE",
    3: "DMA",
}

ENGINE_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "UNIVERSAL",
    2: "COMPUTE",
    3: "EXCLUSIVE_COMPUTE",
    4: "DMA",
    7: "HIGH_PRIORITY_UNIVERSAL",
    8: "HIGH_PRIORITY_GRAPHICS",
}

LOADER_EVENT_TYPE_NAMES = {
    0: "LOAD_TO_GPU_MEMORY",
    1: "UNLOAD_FROM_GPU_MEMORY",
}

ELF_SECTION_TYPE_NAMES = {
    0: "NULL",
    1: "PROGBITS",
    2: "SYMTAB",
    3: "STRTAB",
    4: "RELA",
    5: "HASH",
    6: "DYNAMIC",
    7: "NOTE",
    8: "NOBITS",
    9: "REL",
    11: "DYNSYM",
}


@dataclass
class ChunkHeader:
    offset: int
    type_id: int
    type_name: str
    index: int
    major_version: int
    minor_version: int
    size_in_bytes: int


def _read_c_string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _extract_strings(blob: bytes, min_length: int = 4, limit: int = 20) -> list[str]:
    strings: list[str] = []
    current = bytearray()
    for byte in blob:
        if 32 <= byte <= 126:
            current.append(byte)
        else:
            if len(current) >= min_length:
                strings.append(current.decode("ascii", errors="replace"))
                if len(strings) >= limit:
                    return strings
            current.clear()
    if len(current) >= min_length and len(strings) < limit:
        strings.append(current.decode("ascii", errors="replace"))
    return strings


def _read_pascal_blob(blob: bytes, offset: int, size: int) -> bytes:
    end = min(len(blob), offset + size)
    return blob[offset:end]


def _parse_elf(code: bytes) -> dict[str, Any]:
    if len(code) < 64 or code[:4] != b"\x7fELF":
        return {"valid_elf": False}
    if code[4] != 2 or code[5] != 1:
        return {"valid_elf": False, "reason": "unsupported_elf_class_or_endianness"}

    (
        _e_type,
        _e_machine,
        _e_version,
        e_entry,
        e_phoff,
        e_shoff,
        e_flags,
        _e_ehsize,
        _e_phentsize,
        _e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", code, 16)

    sections = []
    raw_sections = []
    for index in range(e_shnum):
        sh_offset = e_shoff + index * e_shentsize
        if sh_offset + 64 > len(code):
            break
        raw = struct.unpack_from("<IIQQQQIIQQ", code, sh_offset)
        raw_sections.append(raw)

    section_name_table = b""
    if 0 <= e_shstrndx < len(raw_sections):
        sh = raw_sections[e_shstrndx]
        section_name_table = _read_pascal_blob(code, sh[4], sh[5])

    def section_name(name_offset: int) -> str:
        if not section_name_table or name_offset >= len(section_name_table):
            return ""
        return _read_c_string(section_name_table[name_offset:])

    for index, raw in enumerate(raw_sections):
        (
            sh_name,
            sh_type,
            sh_flags,
            sh_addr,
            sh_offset,
            sh_size,
            sh_link,
            sh_info,
            sh_addralign,
            sh_entsize,
        ) = raw
        sections.append(
            {
                "index": index,
                "name": section_name(sh_name),
                "type": sh_type,
                "type_name": ELF_SECTION_TYPE_NAMES.get(sh_type, f"UNKNOWN_{sh_type}"),
                "flags": sh_flags,
                "addr": sh_addr,
                "offset": sh_offset,
                "size": sh_size,
                "link": sh_link,
                "info": sh_info,
                "addralign": sh_addralign,
                "entsize": sh_entsize,
            }
        )

    symbols = []
    section_bytes = {
        section["index"]: _read_pascal_blob(code, section["offset"], section["size"]) for section in sections
    }
    for section in sections:
        if section["type_name"] not in {"SYMTAB", "DYNSYM"}:
            continue
        if section["entsize"] == 0 or section["link"] not in section_bytes:
            continue
        string_table = section_bytes[section["link"]]
        symbol_blob = section_bytes[section["index"]]
        count = min(len(symbol_blob) // section["entsize"], 128)
        for idx in range(count):
            sym_offset = idx * section["entsize"]
            if sym_offset + 24 > len(symbol_blob):
                break
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", symbol_blob, sym_offset
            )
            name = _read_c_string(string_table[st_name:]) if st_name < len(string_table) else ""
            if name:
                symbols.append(
                    {
                        "name": name,
                        "info": st_info,
                        "other": st_other,
                        "shndx": st_shndx,
                        "value": st_value,
                        "size": st_size,
                    }
                )

    notes = []
    for section in sections:
        if section["type_name"] != "NOTE":
            continue
        note_blob = section_bytes[section["index"]]
        cursor = 0
        while cursor + 12 <= len(note_blob):
            namesz, descsz, note_type = struct.unpack_from("<III", note_blob, cursor)
            cursor += 12
            name_blob = _read_pascal_blob(note_blob, cursor, namesz)
            cursor += (namesz + 3) & ~3
            desc_blob = _read_pascal_blob(note_blob, cursor, descsz)
            cursor += (descsz + 3) & ~3
            notes.append(
                {
                    "name": _read_c_string(name_blob),
                    "type": note_type,
                    "desc_size": len(desc_blob),
                }
            )
            if len(notes) >= 32:
                break

    return {
        "valid_elf": True,
        "entry": e_entry,
        "program_header_offset": e_phoff,
        "section_header_offset": e_shoff,
        "flags": e_flags,
        "required_size": e_shoff + e_shentsize * e_shnum,
        "section_count": len(sections),
        "sections": sections,
        "symbols": symbols[:64],
        "notes": notes,
    }


def materialize_code_object_payload(report: dict[str, Any], record: dict[str, Any], pad_elf: bool = True) -> bytes:
    blob = report["_blob"]
    payload_offset = record["payload_offset"]
    payload_size = record["payload_size"]
    payload = bytearray(blob[payload_offset : payload_offset + payload_size])
    elf = record.get("elf", {})
    if pad_elf and elf.get("valid_elf"):
        required_size = int(elf.get("required_size", len(payload)))
        if len(payload) < required_size:
            payload.extend(b"\x00" * (required_size - len(payload)))
    return bytes(payload)


def parse_file_header(blob: bytes) -> dict[str, Any]:
    if len(blob) < 56:
        raise ValueError("file too small for sqtt_file_header")

    fields = struct.unpack_from("<IIIIiiiiiiiiii", blob, 0)
    (
        magic,
        version_major,
        version_minor,
        flags,
        chunk_offset,
        second,
        minute,
        hour,
        day_in_month,
        month,
        year,
        day_in_week,
        day_in_year,
        is_daylight_savings,
    ) = fields

    return {
        "magic": magic,
        "magic_hex": f"0x{magic:08x}",
        "valid_magic": magic == SQTT_FILE_MAGIC_NUMBER,
        "version_major": version_major,
        "version_minor": version_minor,
        "flags": flags,
        "chunk_offset": chunk_offset,
        "timestamp": {
            "year": year + 1900,
            "month": month + 1,
            "day": day_in_month,
            "hour": hour,
            "minute": minute,
            "second": second,
            "day_in_week": day_in_week,
            "day_in_year": day_in_year,
            "is_daylight_savings": is_daylight_savings,
        },
    }


def parse_chunk_header(blob: bytes, offset: int) -> ChunkHeader:
    if offset + 16 > len(blob):
        raise ValueError(f"truncated chunk header at offset {offset}")

    chunk_id_raw, minor_version, major_version, size_in_bytes, _padding = struct.unpack_from(
        "<IHHii", blob, offset
    )
    type_id = chunk_id_raw & 0xFF
    index = (chunk_id_raw >> 8) & 0xFF

    return ChunkHeader(
        offset=offset,
        type_id=type_id,
        type_name=CHUNK_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}"),
        index=index,
        major_version=major_version,
        minor_version=minor_version,
        size_in_bytes=size_in_bytes,
    )


def _parse_queue_event_timings(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    (
        queue_info_record_count,
        queue_info_table_size,
        queue_event_record_count,
        queue_event_table_size,
    ) = struct.unpack_from("<IIII", blob, body_offset)

    cursor = body_offset + 16
    queue_infos = []
    for _ in range(queue_info_record_count):
        queue_id, queue_context, hardware_info, reserved = struct.unpack_from("<QQII", blob, cursor)
        queue_type = hardware_info & 0xFF
        engine_type = (hardware_info >> 8) & 0xFF
        queue_infos.append(
            {
                "queue_id": queue_id,
                "queue_context": queue_context,
                "hardware_info_raw": hardware_info,
                "queue_type": queue_type,
                "queue_type_name": QUEUE_TYPE_NAMES.get(queue_type, f"UNKNOWN_{queue_type}"),
                "engine_type": engine_type,
                "engine_type_name": ENGINE_TYPE_NAMES.get(engine_type, f"UNKNOWN_{engine_type}"),
                "reserved": reserved,
            }
        )
        cursor += 24

    queue_events = []
    for _ in range(queue_event_record_count):
        (
            event_type,
            sqtt_cb_id,
            frame_index,
            queue_info_index,
            submit_sub_index,
            api_id,
            cpu_timestamp,
            gpu_timestamp_0,
            gpu_timestamp_1,
        ) = struct.unpack_from("<IIQIIQQQQ", blob, cursor)
        queue_events.append(
            {
                "event_type": event_type,
                "event_name": QUEUE_EVENT_TYPE_NAMES.get(event_type, f"UNKNOWN_{event_type}"),
                "sqtt_cb_id": sqtt_cb_id,
                "frame_index": frame_index,
                "queue_info_index": queue_info_index,
                "submit_sub_index": submit_sub_index,
                "api_id": api_id,
                "cpu_timestamp": cpu_timestamp,
                "gpu_timestamp_0": gpu_timestamp_0,
                "gpu_timestamp_1": gpu_timestamp_1,
                "gpu_duration": gpu_timestamp_1 - gpu_timestamp_0,
            }
        )
        cursor += 56

    return {
        "queue_info_record_count": queue_info_record_count,
        "queue_info_table_size": queue_info_table_size,
        "queue_event_record_count": queue_event_record_count,
        "queue_event_table_size": queue_event_table_size,
        "queue_infos": queue_infos,
        "queue_events": queue_events,
    }


def _parse_clock_calibration(blob: bytes, chunk: ChunkHeader) -> dict[str, int]:
    body_offset = chunk.offset + 16
    cpu_timestamp, gpu_timestamp, reserved = struct.unpack_from("<QQQ", blob, body_offset)
    return {
        "cpu_timestamp": cpu_timestamp,
        "gpu_timestamp": gpu_timestamp,
        "reserved": reserved,
    }


def _parse_code_object_database(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    offset, flags, size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    payload_start = body_offset + 16
    payload_end = min(chunk.offset + chunk.size_in_bytes, payload_start + size)
    cursor = payload_start
    records = []
    for index in range(record_count):
        if cursor + 4 > payload_end:
            break
        (payload_size,) = struct.unpack_from("<I", blob, cursor)
        record_size = 4 + payload_size
        if payload_size == 0 or cursor + record_size > payload_end:
            break
        record_blob = blob[cursor + 4 : cursor + record_size]
        elf = _parse_elf(record_blob)
        records.append(
            {
                "index": index,
                "record_size": record_size,
                "payload_size": payload_size,
                "record_offset": cursor,
                "payload_offset": cursor + 4,
                "embedded_strings": _extract_strings(record_blob),
                "elf": elf,
            }
        )
        cursor += record_size

    return {
        "offset": offset,
        "flags": flags,
        "size": size,
        "record_count": record_count,
        "records_parsed": len(records),
        "records": records,
    }


def _parse_loader_events(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    offset, flags, record_size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    cursor = body_offset + 16
    records = []
    for index in range(record_count):
        if cursor + record_size > chunk.offset + chunk.size_in_bytes or record_size < 40:
            break
        loader_event_type, reserved, base_address, hash_lo, hash_hi, timestamp = struct.unpack_from(
            "<IIQQQQ", blob, cursor
        )
        records.append(
            {
                "index": index,
                "loader_event_type": loader_event_type,
                "loader_event_type_name": LOADER_EVENT_TYPE_NAMES.get(
                    loader_event_type, f"UNKNOWN_{loader_event_type}"
                ),
                "reserved": reserved,
                "base_address": base_address,
                "code_object_hash": [hash_lo, hash_hi],
                "time_stamp": timestamp,
            }
        )
        cursor += record_size

    return {
        "offset": offset,
        "flags": flags,
        "record_size": record_size,
        "record_count": record_count,
        "records_parsed": len(records),
        "records": records,
    }


def _parse_pso_correlation(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    offset, flags, record_size, record_count = struct.unpack_from("<IIII", blob, body_offset)
    cursor = body_offset + 16
    records = []
    for index in range(record_count):
        if cursor + record_size > chunk.offset + chunk.size_in_bytes or record_size < 88:
            break
        api_pso_hash, hash_lo, hash_hi = struct.unpack_from("<QQQ", blob, cursor)
        name = _read_c_string(blob[cursor + 24 : cursor + 88])
        records.append(
            {
                "index": index,
                "api_pso_hash": api_pso_hash,
                "pipeline_hash": [hash_lo, hash_hi],
                "api_level_obj_name": name,
            }
        )
        cursor += record_size

    return {
        "offset": offset,
        "flags": flags,
        "record_size": record_size,
        "record_count": record_count,
        "records_parsed": len(records),
        "records": records,
    }


def _parse_sqtt_desc(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    shader_engine_index, sqtt_version, spec_version, api_version, compute_unit_index = struct.unpack_from(
        "<iihhi", blob, body_offset
    )
    return {
        "shader_engine_index": shader_engine_index,
        "sqtt_version": sqtt_version,
        "instrumentation_spec_version": spec_version,
        "instrumentation_api_version": api_version,
        "compute_unit_index": compute_unit_index,
    }


def _parse_sqtt_data(blob: bytes, chunk: ChunkHeader) -> dict[str, Any]:
    body_offset = chunk.offset + 16
    offset, size = struct.unpack_from("<ii", blob, body_offset)
    payload_offset = offset
    return {
        "offset": offset,
        "size": size,
        "payload_offset": payload_offset,
        "payload_end": payload_offset + size,
    }


def parse_rgp(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    header = parse_file_header(blob)

    chunks: list[dict[str, Any]] = []
    queue_event_chunks: list[dict[str, Any]] = []
    clock_calibrations: list[dict[str, int]] = []
    code_object_databases: list[dict[str, Any]] = []
    loader_events: list[dict[str, Any]] = []
    pso_correlations: list[dict[str, Any]] = []
    sqtt_descs: list[dict[str, Any]] = []
    sqtt_data_chunks: list[dict[str, Any]] = []

    offset = int(header["chunk_offset"])
    while offset + 16 <= len(blob):
        chunk = parse_chunk_header(blob, offset)
        if chunk.size_in_bytes <= 0:
            raise ValueError(f"invalid chunk size {chunk.size_in_bytes} at offset {offset}")

        chunk_dict = asdict(chunk)
        chunks.append(chunk_dict)

        if chunk.type_name == "QUEUE_EVENT_TIMINGS":
            queue_event_chunks.append(_parse_queue_event_timings(blob, chunk))
        elif chunk.type_name == "CLOCK_CALIBRATION":
            clock_calibrations.append(_parse_clock_calibration(blob, chunk))
        elif chunk.type_name == "CODE_OBJECT_DATABASE":
            code_object_databases.append(_parse_code_object_database(blob, chunk))
        elif chunk.type_name == "CODE_OBJECT_LOADER_EVENTS":
            loader_events.append(_parse_loader_events(blob, chunk))
        elif chunk.type_name == "PSO_CORRELATION":
            pso_correlations.append(_parse_pso_correlation(blob, chunk))
        elif chunk.type_name == "SQTT_DESC":
            sqtt_descs.append(_parse_sqtt_desc(blob, chunk))
        elif chunk.type_name == "SQTT_DATA":
            sqtt_data_chunks.append(_parse_sqtt_data(blob, chunk))

        offset += chunk.size_in_bytes

    return {
        "file": str(path),
        "size_bytes": len(blob),
        "header": header,
        "chunks": chunks,
        "queue_event_chunks": queue_event_chunks,
        "clock_calibrations": clock_calibrations,
        "code_object_databases": code_object_databases,
        "loader_events": loader_events,
        "pso_correlations": pso_correlations,
        "sqtt_descs": sqtt_descs,
        "sqtt_data_chunks": sqtt_data_chunks,
        "_blob": blob,
    }
