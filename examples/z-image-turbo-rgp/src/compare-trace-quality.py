#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_triage(repo_root: Path, capture: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "rgp_analyzer_cli",
        "shader-triage",
        str(capture),
        "--build-helper",
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def trace_quality(triage: dict) -> dict:
    return ((triage.get("summary") or {}).get("trace_quality") or {})


def decoder_summary(triage: dict) -> dict:
    return ((triage.get("summary") or {}).get("decoder") or {})


def profiling_constraints(triage: dict) -> dict:
    return ((triage.get("summary") or {}).get("profiling_constraints") or {})


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} <reference.rgp> <candidate.rgp>", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[3]
    reference = Path(sys.argv[1]).resolve()
    candidate = Path(sys.argv[2]).resolve()

    ref_triage = run_triage(repo_root, reference)
    cand_triage = run_triage(repo_root, candidate)
    ref_quality = trace_quality(ref_triage)
    cand_quality = trace_quality(cand_triage)
    ref_decoder = decoder_summary(ref_triage)
    cand_decoder = decoder_summary(cand_triage)
    ref_constraints = profiling_constraints(ref_triage)
    cand_constraints = profiling_constraints(cand_triage)

    print("trace_quality_compare:")
    print(f"  reference: {reference}")
    print(
        "    "
        f"level={ref_quality.get('runtime_evidence_level')} "
        f"sqtt_bytes={ref_quality.get('sqtt_trace_bytes')} "
        f"instructions={ref_quality.get('decoded_instruction_count')} "
        f"dispatch_spans={ref_quality.get('dispatch_span_count')} "
        f"mapped_dispatch={ref_quality.get('mapped_dispatch_count')}/{ref_quality.get('total_dispatch_count')}"
    )
    print(f"  candidate: {candidate}")
    print(
        "    "
        f"level={cand_quality.get('runtime_evidence_level')} "
        f"sqtt_bytes={cand_quality.get('sqtt_trace_bytes')} "
        f"instructions={cand_quality.get('decoded_instruction_count')} "
        f"dispatch_spans={cand_quality.get('dispatch_span_count')} "
        f"mapped_dispatch={cand_quality.get('mapped_dispatch_count')}/{cand_quality.get('total_dispatch_count')}"
    )
    print("  delta:")
    ref_bytes = int(ref_quality.get("sqtt_trace_bytes") or 0)
    cand_bytes = int(cand_quality.get("sqtt_trace_bytes") or 0)
    ref_inst = int(ref_quality.get("decoded_instruction_count") or 0)
    cand_inst = int(cand_quality.get("decoded_instruction_count") or 0)
    print(f"    sqtt_bytes_ratio={(cand_bytes / ref_bytes):.4f}" if ref_bytes else "    sqtt_bytes_ratio=-")
    print(f"    instruction_ratio={(cand_inst / ref_inst):.4f}" if ref_inst else "    instruction_ratio=-")
    print(
        "  decoder:"
        f" reference_sparse={ref_decoder.get('sparse_runtime_trace')}"
        f" candidate_sparse={cand_decoder.get('sparse_runtime_trace')}"
    )
    print("  profiling_constraints:")
    print(
        "    "
        f"reference_submit_dilution={ref_constraints.get('submit_dilution_suspected')} "
        f"reference_reason={ref_constraints.get('reason')}"
    )
    print(
        "    "
        f"candidate_submit_dilution={cand_constraints.get('submit_dilution_suspected')} "
        f"candidate_reason={cand_constraints.get('reason')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
