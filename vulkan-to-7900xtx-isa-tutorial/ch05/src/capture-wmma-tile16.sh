#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${script_dir}/build"
binary="${build_dir}/ch05_vulkan_gemm"
capture_dir="${chapter_dir}/captures"
repeats="${1:-64}"
mkdir -p "${capture_dir}"
bash "${script_dir}/build.sh" >/dev/null
sleep 1
rm -f /tmp/ch05_vulkan_gemm_*.rgp
env MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE=$((64*1024*1024)) RADV_CACHE_COUNTERS_BUFFER_SIZE=$((64*1024*1024)) \
  "${binary}" "${build_dir}/shaders/wmma_tile16.comp.spv" "wmma_tile16" "${repeats}"
latest="$(ls -1t /tmp/ch05_vulkan_gemm_*.rgp 2>/dev/null | head -n1)"
[[ -n "${latest}" ]] || { echo "no RGP capture found" >&2; exit 1; }
cp "${latest}" "${capture_dir}/wmma_tile16.rgp"
echo "captured ${capture_dir}/wmma_tile16.rgp size_bytes=$(stat -c%s "${capture_dir}/wmma_tile16.rgp")"
