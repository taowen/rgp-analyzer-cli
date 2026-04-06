#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
venv_dir="${repo_root}/vulkan-to-7900xtx-isa-tutorial/ch12/.venv"
output_dir="${chapter_dir}/output"
text="${1:-hello world}"

mkdir -p "${output_dir}"
bash "${script_dir}/setup-venv.sh" >/dev/null
source "${venv_dir}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MANGOHUD=0

python3 "${script_dir}/export-runtime-assets.py" \
  --text "${text}" \
  --output-dir "${output_dir}" \
  --write-gguf-direct

bash "${script_dir}/build.sh" >/dev/null
# direct Python->GGUF is the primary path now; keep C++ packer as fallback/debug tool
if [ ! -f "${output_dir}/ch13-runtime.gguf" ]; then
  "${script_dir}/build/ch13_pack_runtime_gguf" "${output_dir}" "${output_dir}/ch13-runtime.gguf"
fi
"${script_dir}/build/ch13_inspect_runtime_gguf" "${output_dir}/ch13-runtime.gguf"
"${script_dir}/build/ch13_runtime_gguf_smoke" "${output_dir}/ch13-runtime.gguf"
"${script_dir}/build/ch13_runtime_gguf_first_graph" "${output_dir}/ch13-runtime.gguf"
"${script_dir}/build/ch13_runtime_gguf_layer0" "${output_dir}/ch13-runtime.gguf" "${output_dir}/iterative_decode/cond/layer0_refs"
"${script_dir}/build/ch13_runtime_gguf_backbone" "${output_dir}/ch13-runtime.gguf" "${output_dir}/iterative_decode/cond/backbone_refs" gpu_fused_qk
"${script_dir}/build/ch13_runtime_gguf_backbone_tail_hybrid" "${output_dir}/ch13-runtime.gguf" "${output_dir}/iterative_decode/cond/backbone_refs" 0 gpu_fused_qk
for probe_layer in 19 20 21 22 23 24 25 26 27; do
  "${script_dir}/build/ch13_runtime_gguf_layer_probe" \
    "${output_dir}/ch13-runtime.gguf" \
    "${output_dir}/iterative_decode/cond/backbone_refs/layer_$(printf '%02d' "${probe_layer}")_probe" \
    "${probe_layer}"
done
"${script_dir}/build/ch13_runtime_gguf_attn_fused_qk_probe" \
  "${output_dir}/ch13-runtime.gguf" \
  "${output_dir}/iterative_decode/cond/backbone_refs/layer_23_probe" \
  23
"${script_dir}/build/ch13_runtime_gguf_layer_fused_qk_probe" \
  "${output_dir}/ch13-runtime.gguf" \
  "${output_dir}/iterative_decode/cond/backbone_refs/layer_23_probe" \
  23
"${script_dir}/build/ch13_runtime_gguf_iterative_decode" \
  "${output_dir}/ch13-runtime.gguf" \
  "${output_dir}" \
  "${output_dir}/generated_tokens_i32.bin"
