#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shader_dir="${script_dir}/shaders"
output_dir="${script_dir}/build/shaders"
if ! command -v glslc >/dev/null 2>&1; then
  echo "glslc not found." >&2
  exit 1
fi
mkdir -p "${output_dir}"
for shader in wmma_tile16.comp wmma_row2.comp wmma_k2.comp; do
  glslc -fshader-stage=compute "${shader_dir}/${shader}" -o "${output_dir}/${shader}.spv"
  echo "compiled ${output_dir}/${shader}.spv"
done
