#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
whisper_dir="${repo_root}/third_party/whisper.cpp"
threads="${1:-4}"
what="${2:-0}"

bash "${script_dir}/build.sh" >/dev/null
bash "${script_dir}/download-model.sh" tiny.en >/dev/null

cd "${whisper_dir}"
./build-vk/bin/whisper-bench \
  -m ./models/ggml-tiny.en.bin \
  -t "${threads}" \
  -w "${what}"
