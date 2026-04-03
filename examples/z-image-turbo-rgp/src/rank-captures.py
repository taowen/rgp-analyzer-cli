#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


LEVEL_SCORE = {
    "dispatch_isa": 4,
    "decoded_runtime": 3,
    "marker_only": 2,
    "resource_only": 1,
    "no_sqtt": 0,
}


def parse_manifest(path: Path) -> list[Path]:
    captures: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("direct-txt2img_") or " size_bytes=" not in line:
            continue
        name = line.split(" size_bytes=", 1)[0]
        captures.append(path.parent / name)
    return captures


def run_triage(repo_root: Path, capture: Path) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rgp_analyzer_cli",
            "shader-triage",
            str(capture),
            "--build-helper",
            "--json",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def load_phase_markers(capture_dir: Path) -> dict[str, int]:
    phase_dir = capture_dir.parent / "phase-markers"
    markers: dict[str, int] = {}
    for name in (
        "phase_condition_begin",
        "phase_condition_end",
        "phase_diffusion_begin",
        "phase_diffusion_end",
        "phase_vae_begin",
        "phase_vae_end",
    ):
        path = phase_dir / f"{name}.txt"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            markers[name] = int(text)
    return markers


def nearest_phase(markers: dict[str, int], capture: Path) -> tuple[str | None, float | None]:
    if not markers or not capture.is_file():
        return None, None
    capture_ns = capture.stat().st_mtime_ns
    best_name = None
    best_delta = None
    for name, timestamp in markers.items():
        delta = abs(capture_ns - timestamp)
        if best_delta is None or delta < best_delta:
            best_name = name
            best_delta = delta
    if best_name is None or best_delta is None:
        return None, None
    return best_name, best_delta / 1_000_000.0


def score_capture(triage: dict) -> tuple[int, int, int, int]:
    summary = triage.get("summary") or {}
    trace_quality = summary.get("trace_quality") or {}
    level = trace_quality.get("runtime_evidence_level")
    return (
        LEVEL_SCORE.get(str(level), -1),
        int(trace_quality.get("mapped_dispatch_count") or 0),
        int(trace_quality.get("decoded_instruction_count") or 0),
        int(trace_quality.get("sqtt_trace_bytes") or 0),
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <manifest.txt>", file=sys.stderr)
        return 2

    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_file():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[3]
    captures = [path for path in parse_manifest(manifest) if path.is_file()]
    if not captures:
        print("no captures found in manifest", file=sys.stderr)
        return 1

    phase_markers = load_phase_markers(manifest.parent)
    rows = []
    for capture in captures:
        triage = run_triage(repo_root, capture)
        summary = triage.get("summary") or {}
        trace_quality = summary.get("trace_quality") or {}
        decoder = summary.get("decoder") or {}
        phase_name, phase_delta_ms = nearest_phase(phase_markers, capture)
        rows.append(
            {
                "capture": str(capture),
                "trace_quality": trace_quality,
                "decoder": decoder,
                "score": score_capture(triage),
                "nearest_phase": phase_name,
                "phase_delta_ms": phase_delta_ms,
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)

    print("capture_ranking:")
    for index, row in enumerate(rows, start=1):
        quality = row["trace_quality"]
        decoder = row["decoder"]
        print(
            f"  {index}. {row['capture']}\n"
            f"     level={quality.get('runtime_evidence_level')} "
            f"mapped_dispatch={quality.get('mapped_dispatch_count')}/{quality.get('total_dispatch_count')} "
            f"instructions={quality.get('decoded_instruction_count')} "
            f"sqtt_bytes={quality.get('sqtt_trace_bytes')} "
            f"sparse={decoder.get('sparse_runtime_trace')} "
            f"nearest_phase={row.get('nearest_phase')} "
            f"phase_delta_ms={row.get('phase_delta_ms')}"
        )

    best = rows[0]
    print(f"best_capture: {best['capture']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
