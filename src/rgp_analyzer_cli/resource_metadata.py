from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .parser import materialize_code_object_payload


RESOURCE_PATTERNS = {
    "entry_point": re.compile(r"\.entry_point:\s+(\S+)"),
    "lds_size": re.compile(r"\.lds_size:\s+(\d+)"),
    "scratch_memory_size": re.compile(r"\.scratch_memory_size:\s+(\d+)"),
    "sgpr_count": re.compile(r"\.sgpr_count:\s+(\d+)"),
    "vgpr_count": re.compile(r"\.vgpr_count:\s+(\d+)"),
    "wavefront_size": re.compile(r"\.wavefront_size:\s+(\d+)"),
    "spill_threshold": re.compile(r"\.spill_threshold:\s+(\d+)"),
    "user_data_limit": re.compile(r"\.user_data_limit:\s+(\d+)"),
    "api": re.compile(r"\.api:\s+(\S+)"),
}

PIPELINE_HASH_PATTERN = re.compile(
    r"\.internal_pipeline_hash:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)",
    re.MULTILINE,
)
API_SHADER_HASH_PATTERN = re.compile(
    r"\.api_shader_hash:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)",
    re.MULTILINE,
)


def _pick_readelf(tool: str | None = None) -> str:
    candidates = [tool] if tool else []
    candidates.extend(["llvm-readelf-18", "llvm-readelf", "readelf"])
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    raise RuntimeError("llvm-readelf/readelf tool not found")


def _collect_code_object_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for db in report["code_object_databases"]:
        records.extend(db["records"])
    return records


def extract_resource_metadata(report: dict[str, Any], tool: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    readelf = _pick_readelf(tool)
    records = _collect_code_object_records(report)[:limit]
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="rgp-analyzer-note-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for record in records:
            payload = materialize_code_object_payload(report, record, pad_elf=True)
            path = tmpdir_path / f"code-object-{record['index']:03d}.elf"
            path.write_bytes(payload)
            proc = subprocess.run([readelf, "-n", str(path)], capture_output=True, text=True)
            parsed: dict[str, Any] = {
                "index": record["index"],
                "tool": readelf,
                "returncode": proc.returncode,
                "entry_point": None,
                "lds_size": None,
                "scratch_memory_size": None,
                "sgpr_count": None,
                "vgpr_count": None,
                "wavefront_size": None,
                "spill_threshold": None,
                "user_data_limit": None,
                "api": None,
                "internal_pipeline_hash": None,
                "api_shader_hash": None,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            for key, pattern in RESOURCE_PATTERNS.items():
                match = pattern.search(proc.stdout)
                if not match:
                    continue
                value = match.group(1)
                parsed[key] = int(value) if value.isdigit() else value
            pipeline_hash_match = PIPELINE_HASH_PATTERN.search(proc.stdout)
            if pipeline_hash_match:
                parsed["internal_pipeline_hash"] = [int(pipeline_hash_match.group(1)), int(pipeline_hash_match.group(2))]
            api_shader_hash_match = API_SHADER_HASH_PATTERN.search(proc.stdout)
            if api_shader_hash_match:
                parsed["api_shader_hash"] = [int(api_shader_hash_match.group(1)), int(api_shader_hash_match.group(2))]
            results.append(parsed)

    return results
