#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
source_file="${script_dir}/main.cpp"
output_bin="${build_dir}/ch06_vulkan_bank_conflict"
if ! command -v c++ >/dev/null 2>&1; then
  echo "c++ not found." >&2
  exit 1
fi
bash "${script_dir}/compile-shaders.sh" >/dev/null
mkdir -p "${build_dir}"
if pkg-config --exists vulkan 2>/dev/null; then
  read -r -a vulkan_flags <<< "$(pkg-config --cflags --libs vulkan)"
else
  vulkan_flags=(-lvulkan)
fi
tmp_output="$(mktemp "${build_dir}/ch06_vulkan_bank_conflict.XXXXXX")"
trap 'rm -f "${tmp_output}"' EXIT
c++ -std=c++17 -O2 -Wall -Wextra "${source_file}" "${vulkan_flags[@]}" -o "${tmp_output}"
mv -f "${tmp_output}" "${output_bin}"
trap - EXIT
echo "built ${output_bin}"
