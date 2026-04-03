#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _perf_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    families = payload.get("families") or []
    top_row = rows[0] if rows else {}
    top_family = families[0] if families else {}
    return {
        "total_time_us": float(payload.get("total_time_us") or 0.0),
        "top_op": {
            "name": top_row.get("name"),
            "total_us": float(top_row.get("total_us") or 0.0),
        },
        "top_family": {
            "family": top_family.get("family"),
            "total_us": float(top_family.get("total_us") or 0.0),
        },
    }


def _delta(before: float, after: float) -> dict[str, float]:
    payload = {"before": before, "after": after, "delta": after - before}
    if before:
        payload["delta_ratio"] = (after - before) / before
    return payload


def _run_compare_captures(repo_root: Path, baseline_rgp: Path, candidate_rgp: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "rgp_analyzer_cli",
        "compare-captures",
        str(baseline_rgp),
        str(candidate_rgp),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _interpretation(perf_delta_ratio: float | None, hotspot_delta_ratio: float | None, stall_delta_ratio: float | None) -> list[str]:
    notes: list[str] = []
    if perf_delta_ratio is not None:
        if perf_delta_ratio < -0.01:
            notes.append("candidate is faster in real diffusion perf")
        elif perf_delta_ratio > 0.01:
            notes.append("candidate is slower in real diffusion perf")
        else:
            notes.append("real diffusion perf is effectively flat")
    if hotspot_delta_ratio is not None:
        if hotspot_delta_ratio > 0.05:
            notes.append("runtime hotspot total duration increased")
        elif hotspot_delta_ratio < -0.05:
            notes.append("runtime hotspot total duration decreased")
    if stall_delta_ratio is not None:
        if stall_delta_ratio > 0.05:
            notes.append("hotspot stall per hit increased")
        elif stall_delta_ratio < -0.05:
            notes.append("hotspot stall per hit decreased")
    return notes


def main() -> int:
    if len(sys.argv) != 5:
        print(
            f"usage: {Path(sys.argv[0]).name} <baseline-perf.json> <baseline.rgp> <candidate-perf.json> <candidate.rgp>",
            file=sys.stderr,
        )
        return 2

    baseline_perf_path = Path(sys.argv[1]).resolve()
    baseline_rgp_path = Path(sys.argv[2]).resolve()
    candidate_perf_path = Path(sys.argv[3]).resolve()
    candidate_rgp_path = Path(sys.argv[4]).resolve()
    repo_root = Path(__file__).resolve().parents[3]

    baseline_perf = _perf_summary(_load_json(baseline_perf_path))
    candidate_perf = _perf_summary(_load_json(candidate_perf_path))
    compare_payload = _run_compare_captures(repo_root, baseline_rgp_path, candidate_rgp_path)

    perf_total = _delta(baseline_perf["total_time_us"], candidate_perf["total_time_us"])
    perf_top_op = _delta(
        float(baseline_perf["top_op"]["total_us"]),
        float(candidate_perf["top_op"]["total_us"]),
    )
    perf_top_family = _delta(
        float(baseline_perf["top_family"]["total_us"]),
        float(candidate_perf["top_family"]["total_us"]),
    )

    hotspot_profile_deltas = ((compare_payload.get("hotspot") or {}).get("profile_deltas") or {})
    hotspot_deltas = ((compare_payload.get("hotspot") or {}).get("deltas") or {})
    interpretations = _interpretation(
        perf_total.get("delta_ratio"),
        ((hotspot_deltas.get("total_duration") or {}).get("delta_ratio")),
        ((hotspot_profile_deltas.get("avg_stall_per_hit") or {}).get("delta_ratio")),
    )

    result = {
        "baseline_perf_json": str(baseline_perf_path),
        "baseline_rgp": str(baseline_rgp_path),
        "candidate_perf_json": str(candidate_perf_path),
        "candidate_rgp": str(candidate_rgp_path),
        "perf": {
            "total_time_us": perf_total,
            "top_op": {
                "baseline": baseline_perf["top_op"],
                "candidate": candidate_perf["top_op"],
                "delta": perf_top_op,
            },
            "top_family": {
                "baseline": baseline_perf["top_family"],
                "candidate": candidate_perf["top_family"],
                "delta": perf_top_family,
            },
        },
        "rgp": compare_payload,
        "interpretation": interpretations,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
