#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shader_dir="${script_dir}/shaders"
out_dir="${script_dir}/build/shaders"

mkdir -p "${out_dir}"

for shader in "${shader_dir}"/*.comp; do
    out="${out_dir}/$(basename "${shader%.comp}").spv"
    glslc -fshader-stage=compute --target-env=vulkan1.1 --target-spv=spv1.3 "${shader}" -O -o "${out}"
    echo "compiled ${shader} -> ${out}"
done
