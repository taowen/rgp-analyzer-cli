#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

cmake -S "${script_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF >/dev/null

cmake --build "${build_dir}" -j"$(nproc)" >/dev/null
echo "built ch13 pack/inspect/runtime gguf tools in ${build_dir}"
