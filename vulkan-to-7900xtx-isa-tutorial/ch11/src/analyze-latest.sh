#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
capture="${chapter_dir}/captures/latest.rgp"

if [[ ! -f "${capture}" ]]; then
  echo "capture not found: ${capture}" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src python3 -m rgp_analyzer_cli resource-summary "${capture}"
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-triage "${capture}" --build-helper
echo
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-focus "${capture}"
echo
echo "# flash_attn candidate code_object[2]"
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" \
  --code-object-index 2 \
  --source-file "${repo_root}/third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"
echo
echo "# flash_attn candidate code_object[5]"
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" \
  --code-object-index 5 \
  --source-file "${repo_root}/third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"
