#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
capture="${repo_root}/examples/minimal-vulkan-compute-rgp/captures/latest.rgp"

if [[ ! -f "${capture}" ]]; then
    echo "missing capture: ${capture}" >&2
    exit 1
fi

PYTHONPATH="${repo_root}/src" REPO_ROOT="${repo_root}" CAPTURE="${capture}" python3 - <<'PY'
import os
from pathlib import Path

from rgp_analyzer_cli.models import DecodeOptions, DispatchIsaOptions, TriageOptions
from rgp_analyzer_cli.services import (
    decode_payload,
    dispatch_isa_payload,
    load_capture,
    resource_payload,
    stitch_payload,
    triage_payload,
)

capture = Path(os.environ["CAPTURE"])
session = load_capture(capture)

print("release_check:")

stitch = stitch_payload(session)
decode = decode_payload(session, DecodeOptions(stream_limit=1, hotspot_limit=4, build_helper=True))
resource = resource_payload(session, limit=10)
dispatch = dispatch_isa_payload(session, DispatchIsaOptions(stream_index=0, dispatch_limit=8))
triage = triage_payload(session, TriageOptions(build_helper=True))

stitched = decode.get("stitched") or {}
overview = stitched.get("dispatch_isa_overview") or {}

print(f"  resolved_entries={stitch.get('resolved_entry_count', 0)}")
print(f"  dispatch_spans={stitch.get('dispatch_api_span_count', 0)}")
print(f"  dispatch_isa={overview.get('mapped_dispatch_count', 0)}/{overview.get('dispatch_count', 0)}")
print(f"  resource_code_objects={len(resource) if isinstance(resource, list) else 0}")
print(f"  dispatch_segments={dispatch.get('dispatch_count', 0)}")
print(f"  triage_findings={len(triage.get('findings') or [])}")

assert stitch.get("resolved_entry_count", 0) >= 1
assert overview.get("mapped_dispatch_count", 0) >= 1
assert isinstance(resource, list) and len(resource) >= 1
assert dispatch.get("dispatch_count", 0) >= 1
assert len(triage.get("findings") or []) >= 1
PY
