#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
capture="${chapter_dir}/captures/latest.rgp"
source_file="${script_dir}/../../../third_party/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

if [[ ! -f "${capture}" ]]; then
  echo "capture not found: ${capture}" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src python3 -m rgp_analyzer_cli resource-summary "${capture}"
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-triage "${capture}" --build-helper
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-focus "${capture}" --source-file "${source_file}"
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" --source-file "${source_file}"
