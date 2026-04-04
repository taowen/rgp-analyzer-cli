#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
binary="${build_dir}/ch05_vulkan_gemm"
repeats="${1:-64}"
bash "${script_dir}/build.sh" >/dev/null
"${binary}" "${build_dir}/shaders/wmma_k2.comp.spv" "wmma_k2" "${repeats}"
