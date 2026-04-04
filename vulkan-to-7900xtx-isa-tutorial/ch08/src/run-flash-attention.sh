#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
bash "${script_dir}/compile-shaders.sh" >/dev/null
bash "${script_dir}/build.sh" >/dev/null
exec env MANGOHUD=0 "${build_dir}/ch08_attention" \
    --shader "${build_dir}/shaders/flash_attention.spv" \
    --seq-len 64 --head-dim 64 --dispatches "${1:-64}" --warmup 2 --repeats 8
