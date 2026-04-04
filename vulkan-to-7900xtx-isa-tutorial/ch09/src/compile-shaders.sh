#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shader_dir="${script_dir}/shaders"
out_dir="${script_dir}/build/shaders"
local_glslang="/tmp/glslang/build/StandAlone/glslang"

mkdir -p "${out_dir}"

compile_with_local_glslang=false
if [[ -x "${local_glslang}" ]]; then
    compile_with_local_glslang=true
fi

for shader in "${shader_dir}"/*.comp; do
    out="${out_dir}/$(basename "${shader%.comp}").spv"
    if [[ "${compile_with_local_glslang}" == true ]]; then
        "${local_glslang}" -V --target-env vulkan1.3 "${shader}" -o "${out}"
    else
        glslc -fshader-stage=compute --target-env=vulkan1.1 --target-spv=spv1.3 "${shader}" -O -o "${out}"
    fi
    echo "compiled ${shader} -> ${out}"
done
