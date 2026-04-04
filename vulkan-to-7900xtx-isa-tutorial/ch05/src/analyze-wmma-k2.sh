#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
capture="${chapter_dir}/captures/wmma_k2.rgp"
source_file="${script_dir}/shaders/wmma_k2.comp"
[[ -f "${capture}" ]] || { echo "capture not found: ${capture}" >&2; exit 1; }
analyzer_repo="${RGP_ANALYZER_REPO:-$(cd "${script_dir}/../../.." && pwd)}"
cd "${analyzer_repo}"
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-focus "${capture}" --source-file "${source_file}" --no-cache
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" --source-file "${source_file}" --no-cache
