#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
whisper_dir="${repo_root}/third_party/whisper.cpp"

cd "${whisper_dir}"

echo "=== baseline: -t 4 ==="
./build-vk/bin/whisper-cli \
  -m ./models/ggml-tiny.en.bin \
  -f ./samples/jfk.wav \
  -t 4 | rg "sample time|encode time|decode time|batchd time|total time|^\\[00:"

echo
echo "=== optimized: -t 12 -bo 1 -bs 1 -nf -mc 16 ==="
./build-vk/bin/whisper-cli \
  -m ./models/ggml-tiny.en.bin \
  -f ./samples/jfk.wav \
  -t 12 \
  -bo 1 \
  -bs 1 \
  -nf \
  -mc 16 | rg "sample time|encode time|decode time|batchd time|total time|^\\[00:"
