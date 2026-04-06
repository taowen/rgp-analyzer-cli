#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
whisper_dir="${repo_root}/third_party/whisper.cpp"
build_dir="${whisper_dir}/build-vk"

if [[ ! -d "${whisper_dir}" ]]; then
  echo "missing third_party/whisper.cpp: ${whisper_dir}" >&2
  exit 1
fi

cmake -S "${whisper_dir}" -B "${build_dir}" \
  -DGGML_VULKAN=1 \
  -DWHISPER_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release >/dev/null

cmake --build "${build_dir}" -j"$(nproc)" --config Release >/dev/null
echo "built ${build_dir}/bin/whisper-cli"
echo "built ${build_dir}/bin/whisper-bench"
