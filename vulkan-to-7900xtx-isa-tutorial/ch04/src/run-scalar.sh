#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/build/ch04_vulkan_matmul"
shader="${script_dir}/build/shaders/scalar_matmul.comp.spv"
repeats="${1:-64}"
bash "${script_dir}/build.sh"
"${binary}" "${shader}" scalar "${repeats}"
