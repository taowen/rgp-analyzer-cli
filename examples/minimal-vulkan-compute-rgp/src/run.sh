#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
shader_name="${1:-baseline}"
shift || true

bash "${script_dir}/compile-shaders.sh" >/dev/null
bash "${script_dir}/build.sh"

MANGOHUD=0 "${build_dir}/minimal_compute" \
    --shader "${build_dir}/shaders/${shader_name}.spv" \
    "$@"
