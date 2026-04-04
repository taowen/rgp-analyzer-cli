#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build/shaders"
shader_src="${script_dir}/shaders/fill_buffer.comp"
shader_out="${build_dir}/fill_buffer.comp.spv"

if ! command -v glslc >/dev/null 2>&1; then
    echo "glslc not found. Install glslang-tools or Vulkan SDK." >&2
    exit 1
fi

mkdir -p "${build_dir}"
glslc -O "${shader_src}" -o "${shader_out}"
echo "compiled ${shader_out}"
