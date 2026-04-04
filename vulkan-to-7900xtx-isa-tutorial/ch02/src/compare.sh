#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
baseline_capture="${chapter_dir}/captures/baseline.rgp"
candidate_capture="${chapter_dir}/captures/reg_pressure.rgp"

if [[ ! -f "${baseline_capture}" ]]; then
    echo "capture not found: ${baseline_capture}" >&2
    exit 1
fi
if [[ ! -f "${candidate_capture}" ]]; then
    echo "capture not found: ${candidate_capture}" >&2
    exit 1
fi

if [[ -n "${RGP_ANALYZER_REPO:-}" ]]; then
    analyzer_repo="${RGP_ANALYZER_REPO}"
else
    analyzer_repo="$(cd "${script_dir}/../../.." && pwd)"
fi

cd "${analyzer_repo}"
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-captures "${baseline_capture}" "${candidate_capture}" --no-cache
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline_capture}" "${candidate_capture}" --no-cache
