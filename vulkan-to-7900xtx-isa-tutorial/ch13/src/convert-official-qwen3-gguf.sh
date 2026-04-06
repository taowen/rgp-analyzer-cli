#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
venv_dir="${repo_root}/vulkan-to-7900xtx-isa-tutorial/ch12/.venv"
llama_cpp_dir="${HOME}/projects/llama.cpp"
output_dir="${chapter_dir}/output"
hf_dir="${output_dir}/qwen3-hf-extracted"
gguf_path="${output_dir}/qwen3-from-omnivoice-f16.gguf"

bash "${script_dir}/setup-venv.sh" >/dev/null
source "${venv_dir}/bin/activate"

python3 "${script_dir}/extract-official-qwen3-hf.py" \
  --output-dir "${hf_dir}"

python3 "${llama_cpp_dir}/convert_hf_to_gguf.py" \
  "${hf_dir}" \
  --outfile "${gguf_path}" \
  --outtype f16

echo "official_qwen3_gguf=${gguf_path}"
