#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${chapter_dir}/output"

bash "${script_dir}/build.sh" >/dev/null

layer="${1:-23}"
probe_dir="${output_dir}/iterative_decode/cond/backbone_refs/layer_$(printf '%02d' "${layer}")_probe"
gguf_path="${output_dir}/ch13-runtime.gguf"

echo "=== CPU ==="
"${script_dir}/build/ch13_runtime_gguf_layer_probe" \
  "${gguf_path}" \
  "${probe_dir}" \
  "${layer}" \
  cpu | rg 'probe\.backend=|probe\.(attn_out|o_proj|mlp_act|mlp_down|layer_out)\.(max_abs_diff|mean_abs_diff)|layer_probe_ok'

echo
echo "=== Vulkan ==="
"${script_dir}/build/ch13_runtime_gguf_layer_probe" \
  "${gguf_path}" \
  "${probe_dir}" \
  "${layer}" \
  gpu | rg 'probe\.backend=|probe\.(attn_out|o_proj|mlp_act|mlp_down|layer_out)\.(max_abs_diff|mean_abs_diff)|layer_probe_ok'
