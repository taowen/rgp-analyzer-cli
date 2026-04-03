from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .decode_bridge import decoder_bridge, find_code_object_record
from .parser import materialize_code_object_payload
from .sqtt_stitch import stitch_hotspots


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_native_helper(repo_root_path: Path) -> Path:
    build_script = repo_root_path / "native" / "rocm_sqtt_decoder_helper" / "build.sh"
    proc = subprocess.run([str(build_script)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to build helper\n{proc.stdout}\n{proc.stderr}")
    helper = repo_root_path / "native" / "rocm_sqtt_decoder_helper" / "build" / "rocm-sqtt-decoder-helper"
    if not helper.exists():
        raise RuntimeError(f"helper build succeeded but executable was not found at {helper}")
    return helper


def default_decoder_lib_dir() -> Path | None:
    candidates = [
        Path("/home/taowen/projects/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"),
        Path("/opt/rocm/lib"),
    ]
    for candidate in candidates:
        if (candidate / "librocprof-trace-decoder.so").exists():
            return candidate
    return None


def resolve_helper_path(repo_root_path: Path, helper: Path | None, build_helper_flag: bool) -> Path:
    resolved = helper or (repo_root_path / "native" / "rocm_sqtt_decoder_helper" / "build" / "rocm-sqtt-decoder-helper")
    if build_helper_flag or not resolved.exists():
        resolved = build_native_helper(repo_root_path)
    return resolved


def prepare_decode_artifacts(
    report: dict[str, Any],
    *,
    temp_root: Path,
    stream_limit: int,
) -> tuple[dict[str, Any], Path]:
    bridge = decoder_bridge(report)
    code_dir = temp_root / "code-objects"
    sqtt_dir = temp_root / "sqtt"
    code_dir.mkdir(parents=True, exist_ok=True)
    sqtt_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# manifest for rocm_sqtt_decoder_helper",
        "# type<TAB>fields...",
    ]

    for item in bridge["entries"]:
        record = find_code_object_record(report, item["code_object_index"])
        if record is None:
            continue
        path = code_dir / f"code-object-{item['bridge_index']:03d}.elf"
        path.write_bytes(materialize_code_object_payload(report, record, pad_elf=True))
        manifest_lines.append(
            "\t".join(
                [
                    "CO",
                    str(item["load_id"]),
                    str(item["load_addr"]),
                    str(item["load_size"]),
                    str(path),
                ]
            )
        )

    count = len(report["sqtt_data_chunks"])
    if stream_limit and stream_limit > 0:
        count = min(count, stream_limit)

    for index in range(count):
        desc = report["sqtt_descs"][index] if index < len(report["sqtt_descs"]) else {}
        chunk = report["sqtt_data_chunks"][index]
        path = sqtt_dir / f"sqtt-{index:03d}.bin"
        path.write_bytes(report["_blob"][chunk["payload_offset"] : chunk["payload_end"]])
        manifest_lines.append(
            "\t".join(
                [
                    "SQTT",
                    str(index),
                    str(desc.get("shader_engine_index", -1)),
                    str(desc.get("compute_unit_index", -1)),
                    str(path),
                ]
            )
        )

    manifest_path = temp_root / "decoder-manifest.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return bridge, manifest_path


def run_decode_helper(
    report: dict[str, Any],
    *,
    helper: Path,
    decoder_lib_dir: Path | None,
    stream_limit: int,
    hotspot_limit: int,
    strict: bool,
    as_json: bool,
    keep_temp: bool,
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="rgp-analyzer-decode-"))
    try:
        bridge, manifest_path = prepare_decode_artifacts(report, temp_root=temp_root, stream_limit=stream_limit)
        cmd = [str(helper), "--manifest", str(manifest_path), "--hotspot-limit", str(hotspot_limit)]
        if decoder_lib_dir is not None:
            cmd.extend(["--decoder-lib-dir", str(decoder_lib_dir)])
        if as_json:
            cmd.append("--json")
        if strict:
            cmd.append("--strict")

        env = dict(os.environ)
        if decoder_lib_dir is not None:
            env["ROCPROFILER_TRACE_DECODER_LIB_PATH"] = str(decoder_lib_dir)
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = str(decoder_lib_dir) if not existing else f"{decoder_lib_dir}:{existing}"

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        parsed_json = None
        if as_json and proc.stdout.strip():
            parsed_json = json.loads(proc.stdout)
            stitched = stitch_hotspots(report, parsed_json)
            if stitched is not None:
                parsed_json["stitched"] = stitched
            parsed_json["bridge"] = bridge
        return {
            "bridge": bridge,
            "manifest_path": manifest_path,
            "temp_root": temp_root,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "json": parsed_json,
            "kept_temp": keep_temp,
        }
    finally:
        if not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
