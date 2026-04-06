#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
output_dir="${chapter_dir}/output"
whisper_dir="${repo_root}/third_party/whisper.cpp"

target_text="${1:-This is a short speech synthesis test running on an AMD Radeon graphics card.}"

mkdir -p "${output_dir}"
bash "${repo_root}/vulkan-to-7900xtx-isa-tutorial/ch11/src/build.sh" >/dev/null
bash "${repo_root}/vulkan-to-7900xtx-isa-tutorial/ch11/src/download-model.sh" tiny.en >/dev/null

cd "${whisper_dir}"
./build-vk/bin/whisper-cli \
  -m ./models/ggml-tiny.en.bin \
  -f "${output_dir}/generated.wav" \
  -otxt -of "${output_dir}/asr" \
  -t 8 >/tmp/ch12-whisper.log 2>&1

cat "${output_dir}/asr.txt"
python3 "${script_dir}/compare_asr.py" \
  --target "${target_text}" \
  --asr-file "${output_dir}/asr.txt" \
  --json-output "${output_dir}/verification.json"
