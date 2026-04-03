#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
capture_path="${repo_root}/examples/medium-vulkan-graph-rgp/captures/latest.rgp"

PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli inspect "${capture_path}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli stitch-report "${capture_path}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage "${capture_path}" --build-helper
