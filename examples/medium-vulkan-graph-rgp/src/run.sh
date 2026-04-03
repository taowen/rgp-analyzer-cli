#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

if [[ ! -f "${build_dir}/shaders/preprocess.spv" || ! -f "${build_dir}/shaders/mix.spv" || ! -f "${build_dir}/shaders/reduce.spv" ]]; then
    bash "${script_dir}/compile-shaders.sh"
fi

if [[ ! -x "${build_dir}/medium_graph_compute" ]]; then
    bash "${script_dir}/build.sh"
fi

"${build_dir}/medium_graph_compute" \
    --preprocess "${build_dir}/shaders/preprocess.spv" \
    --mix "${build_dir}/shaders/mix.spv" \
    --reduce "${build_dir}/shaders/reduce.spv" \
    "$@"
