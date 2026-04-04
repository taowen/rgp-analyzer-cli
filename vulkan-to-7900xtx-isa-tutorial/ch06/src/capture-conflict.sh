#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${script_dir}/build"
binary="${build_dir}/ch06_vulkan_bank_conflict"
capture_dir="${chapter_dir}/captures"
repeats="${1:-128}"
mkdir -p "${capture_dir}"
bash "${script_dir}/build.sh" >/dev/null
sleep 1
rm -f /tmp/ch06_vulkan_bank_conflict_*.rgp
env MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE=$((64*1024*1024)) RADV_CACHE_COUNTERS_BUFFER_SIZE=$((64*1024*1024)) \
  "${binary}" "${build_dir}/shaders/lds_conflict.comp.spv" "lds_conflict" "${repeats}"
latest="$(ls -1t /tmp/ch06_vulkan_bank_conflict_*.rgp 2>/dev/null | head -n1)"
[[ -n "${latest}" ]] || { echo "no RGP capture found" >&2; exit 1; }
cp "${latest}" "${capture_dir}/conflict.rgp"
echo "captured ${capture_dir}/conflict.rgp size_bytes=$(stat -c%s "${capture_dir}/conflict.rgp")"
