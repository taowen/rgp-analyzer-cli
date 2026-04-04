#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
capture_dir="${chapter_dir}/captures"
binary="${script_dir}/build/ch03_vulkan_bottlenecks"
shader="${script_dir}/build/shaders/memory_heavy.comp.spv"
repeats="${1:-128}"
bash "${script_dir}/build.sh" >/dev/null
mkdir -p "${capture_dir}"
before_list="$(mktemp)"
after_list="$(mktemp)"
trap 'rm -f "${before_list}" "${after_list}"' EXIT
find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${before_list}"
env MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE=67108864 "${binary}" "${shader}" memory_heavy "${repeats}"
find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_list}"
mapfile -t new_files < <(comm -13 "${before_list}" "${after_list}")
if [[ "${#new_files[@]}" -eq 0 ]]; then echo "no new .rgp files found in /tmp" >&2; exit 1; fi
latest_path=""
latest_size=-1
for name in "${new_files[@]}"; do
  path="/tmp/${name}"
  [[ -f "${path}" ]] || continue
  size="$(stat -c '%s' "${path}")"
  if (( size > latest_size )); then latest_size="${size}"; latest_path="${path}"; fi
done
[[ -n "${latest_path}" ]] || { echo "failed to choose a capture file" >&2; exit 1; }
cp "${latest_path}" "${capture_dir}/memory_heavy.rgp"
echo "captured ${capture_dir}/memory_heavy.rgp size_bytes=${latest_size}"
