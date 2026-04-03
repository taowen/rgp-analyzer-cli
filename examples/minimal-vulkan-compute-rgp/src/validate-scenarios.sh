#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
capture_dir="${example_dir}/captures"

analyze_only=0
skip_decode=0
if [[ "${1:-}" == "--analyze-only" ]]; then
    analyze_only=1
fi
if [[ "${2:-}" == "--skip-decode" || "${1:-}" == "--skip-decode" ]]; then
    skip_decode=1
fi

if [[ "${analyze_only}" -eq 0 ]]; then
    bash "${script_dir}/capture-scenarios.sh"
fi

PYTHONPATH="${repo_root}/src" REPO_ROOT="${repo_root}" SKIP_DECODE="${skip_decode}" python3 - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
skip_decode = os.environ.get("SKIP_DECODE", "0") == "1"
capture_dir = repo_root / "examples" / "minimal-vulkan-compute-rgp" / "captures"
scenarios = [
    ("single-baseline", capture_dir / "single-baseline.rgp"),
    ("multi-pipeline", capture_dir / "multi-pipeline.rgp"),
    ("multi-cmdbuf", capture_dir / "multi-cmdbuf.rgp"),
    ("barrier-mix", capture_dir / "barrier-mix.rgp"),
]

def run_json(*args: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "rgp_analyzer_cli", *args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)

print("scenario_validation:")
for name, capture in scenarios:
    stitch = run_json("stitch-report", str(capture), "--json")
    resource = run_json("resource-summary", str(capture), "--json")
    overview = {}
    if not skip_decode:
        decode = run_json("decode-sqtt", str(capture), "--build-helper", "--json")
        stitched = decode.get("stitched") or {}
        overview = stitched.get("dispatch_isa_overview") or {}
    top_resource = resource[0] if isinstance(resource, list) and resource else {}
    print(
        f"  {name}: "
        f"resolved={stitch.get('resolved_entry_count', 0)} "
        f"bind={stitch.get('bind_marker_count', 0)} "
        f"dispatch_spans={stitch.get('dispatch_api_span_count', 0)} "
        f"cb_spans={stitch.get('command_buffer_span_count', 0)} "
        f"dispatch_isa={overview.get('mapped_dispatch_count', 0)}/{overview.get('dispatch_count', 0)} "
        f"vgpr={top_resource.get('vgpr_count', 'n/a')} "
        f"lds={top_resource.get('lds_size', 'n/a')}"
    )
    for item in (overview.get("ordered") or [])[:3]:
        top_pc = item.get("top_pc") or {}
        top_pc_text = ""
        if top_pc:
            top_pc_text = (
                f" top_pc=0x{int(top_pc.get('pc', 0)):x} "
                f"{top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
            )
        share = item.get("mapped_dispatch_share")
        share_text = f" share={share:.2f}" if isinstance(share, (int, float)) else ""
        print(
            f"    code_object[{item['code_object_index']}] "
            f"mapped={item.get('mapped_dispatch_count', 0)}/{item.get('dispatch_count', 0)}"
            f"{share_text}{top_pc_text}"
        )
PY
