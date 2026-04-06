#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
capture="${chapter_dir}/captures/latest.rgp"
source_file="${repo_root}/third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"

if [[ ! -f "${capture}" ]]; then
  echo "capture not found: ${capture}" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src python3 -m rgp_analyzer_cli code-object-isa "${capture}" \
  --code-object-index 5 \
  --source-file "${source_file}"
