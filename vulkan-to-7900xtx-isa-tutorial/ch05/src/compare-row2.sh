#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
baseline="${chapter_dir}/captures/wmma_tile16.rgp"
candidate="${chapter_dir}/captures/wmma_row2.rgp"
source_file="${script_dir}/shaders/wmma_row2.comp"
analyzer_repo="${RGP_ANALYZER_REPO:-$(cd "${script_dir}/../../.." && pwd)}"
cd "${analyzer_repo}"
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline}" "${candidate}" --source-file "${source_file}" --no-cache
