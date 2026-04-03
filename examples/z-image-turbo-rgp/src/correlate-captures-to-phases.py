#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


PHASE_FILES = [
    "phase_condition_begin.txt",
    "phase_condition_end.txt",
    "phase_diffusion_begin.txt",
    "phase_diffusion_end.txt",
    "phase_vae_begin.txt",
    "phase_vae_end.txt",
]


def load_phase_markers(phase_dir: Path) -> dict[str, int]:
    markers: dict[str, int] = {}
    for name in PHASE_FILES:
        path = phase_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        markers[name.removesuffix(".txt")] = int(text)
    return markers


def load_phase_markers_from_manifest(manifest_path: Path) -> dict[str, int]:
    markers: dict[str, int] = {}
    in_section = False
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "phase_markers:":
            in_section = True
            continue
        if in_section and not line.startswith("  "):
            break
        if not in_section or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name and value:
            markers[name] = int(value)
    return markers


def nearest_phase(markers: dict[str, int], capture_ns: int) -> tuple[str, int] | None:
    if not markers:
        return None
    best_name = None
    best_delta = None
    for name, ts in markers.items():
        delta = abs(capture_ns - ts)
        if best_delta is None or delta < best_delta:
            best_name = name
            best_delta = delta
    if best_name is None or best_delta is None:
        return None
    return best_name, best_delta


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: correlate-captures-to-phases.py <capture-dir>", file=sys.stderr)
        return 2

    capture_dir = Path(sys.argv[1]).resolve()
    phase_dir = capture_dir.parent / "phase-markers"
    manifest_path = capture_dir / "last-capture-manifest.txt"
    markers = load_phase_markers_from_manifest(manifest_path) if manifest_path.is_file() else {}
    if not markers:
        markers = load_phase_markers(phase_dir)

    print("phase_correlation:")
    if not markers:
        print("  phase_markers: none")
        return 0

    print(f"  phase_dir: {phase_dir}")
    for name in PHASE_FILES:
        key = name.removesuffix(".txt")
        if key in markers:
            print(f"  {key}: {markers[key]}")

    captures = sorted(capture_dir.glob("direct-txt2img*.rgp"))
    if not captures:
        print("  captures: none")
        return 0

    print("  captures:")
    for path in captures:
        stat = path.stat()
        capture_ns = stat.st_mtime_ns
        nearest = nearest_phase(markers, capture_ns)
        if nearest is None:
            print(f"    {path.name}: mtime_ns={capture_ns}")
            continue
        phase_name, delta_ns = nearest
        print(
            f"    {path.name}: mtime_ns={capture_ns} nearest_phase={phase_name} delta_ms={delta_ns / 1_000_000:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
