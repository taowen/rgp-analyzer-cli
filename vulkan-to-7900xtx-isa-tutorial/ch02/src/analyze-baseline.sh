#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
capture="${chapter_dir}/captures/baseline.rgp"
source_file="${script_dir}/shaders/baseline.comp"

if [[ ! -f "${capture}" ]]; then
    echo "capture not found: ${capture}" >&2
    exit 1
fi

if [[ -n "${RGP_ANALYZER_REPO:-}" ]]; then
    analyzer_repo="${RGP_ANALYZER_REPO}"
else
    analyzer_repo="$(cd "${script_dir}/../../.." && pwd)"
fi

cd "${analyzer_repo}"
PYTHONPATH=src python3 -m rgp_analyzer_cli resource-summary "${capture}"
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-focus "${capture}" --source-file "${source_file}" --no-cache
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" --source-file "${source_file}" --no-cache
