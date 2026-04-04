#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
binary="${build_dir}/ch06_vulkan_bank_conflict"
repeats="${1:-128}"
bash "${script_dir}/build.sh" >/dev/null
"${binary}" "${build_dir}/shaders/lds_conflict.comp.spv" "lds_conflict" "${repeats}"
