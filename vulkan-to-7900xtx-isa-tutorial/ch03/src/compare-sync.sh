#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
baseline_capture="${chapter_dir}/captures/baseline.rgp"
candidate_capture="${chapter_dir}/captures/sync_heavy.rgp"
[[ -f "${baseline_capture}" ]] || { echo "capture not found: ${baseline_capture}" >&2; exit 1; }
[[ -f "${candidate_capture}" ]] || { echo "capture not found: ${candidate_capture}" >&2; exit 1; }
analyzer_repo="${RGP_ANALYZER_REPO:-$(cd "${script_dir}/../../.." && pwd)}"
cd "${analyzer_repo}"
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline_capture}" "${candidate_capture}" --no-cache
