#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
mkdir -p "${build_dir}"

if pkg-config --exists vulkan 2>/dev/null; then
    read -r -a vulkan_flags <<< "$(pkg-config --cflags --libs vulkan)"
else
    vulkan_flags=(-lvulkan)
fi

c++ -std=c++17 -O2 -Wall -Wextra \
    "${script_dir}/main.cpp" \
    "${vulkan_flags[@]}" \
    -o "${build_dir}/minimal_compute"
