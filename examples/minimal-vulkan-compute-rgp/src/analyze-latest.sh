#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
capture="${repo_root}/examples/minimal-vulkan-compute-rgp/captures/latest.rgp"

if [[ ! -f "${capture}" ]]; then
    echo "missing capture: ${capture}" >&2
    exit 1
fi

PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli inspect "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli resource-summary "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli stitch-report "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli dispatch-isa-map "${capture}" --stream-index 0 --dispatch-limit 8
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli decode-sqtt "${capture}" --build-helper
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage "${capture}" --build-helper
