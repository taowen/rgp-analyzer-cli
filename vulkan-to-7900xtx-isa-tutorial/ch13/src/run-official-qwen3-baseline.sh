#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
llama_cpp_dir="${HOME}/projects/llama.cpp"
output_dir="${chapter_dir}/output"
gguf_path="${output_dir}/qwen3-from-omnivoice-f16.gguf"
prompt="${1:-Hello}"

bash "${script_dir}/convert-official-qwen3-gguf.sh" >/dev/null

cmake -S "${llama_cpp_dir}" -B "${llama_cpp_dir}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON >/dev/null
cmake --build "${llama_cpp_dir}/build" --target llama-cli -j"$(nproc)" >/dev/null

"${llama_cpp_dir}/build/bin/llama-cli" \
  -m "${gguf_path}" \
  -p "${prompt}" \
  -n 32 \
  --no-warmup
