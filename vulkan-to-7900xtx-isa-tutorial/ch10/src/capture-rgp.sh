#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
capture_dir="${chapter_dir}/captures"
binary="${script_dir}/build/ch10_ggml_isa"
repeats="${1:-8}"

bash "${script_dir}/build.sh" >/dev/null
mkdir -p "${capture_dir}"

before_list="$(mktemp)"
after_list="$(mktemp)"
app_log="$(mktemp)"
trap 'rm -f "${before_list}" "${after_list}" "${app_log}"' EXIT

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${before_list}"
sleep 1
env \
  GGML_VK_DISABLE_FUSION=1 \
  GGML_VK_PROFILE_NODES_PER_SUBMIT=1 \
  MESA_VK_TRACE=rgp \
  MESA_VK_TRACE_PER_SUBMIT=1 \
  RADV_THREAD_TRACE_BUFFER_SIZE=67108864 \
  "${binary}" "${repeats}" >"${app_log}" 2>&1 &
app_pid=$!

latest_path=""
latest_size=-1

for _ in $(seq 1 60); do
  find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_list}"
  mapfile -t new_files < <(comm -13 "${before_list}" "${after_list}")
  if [[ "${#new_files[@]}" -gt 0 ]]; then
    for name in "${new_files[@]}"; do
      path="/tmp/${name}"
      [[ -f "${path}" ]] || continue
      size="$(stat -c '%s' "${path}")"
      if (( size > latest_size )); then
        latest_size="${size}"
        latest_path="${path}"
      fi
    done
  fi
  if [[ -n "${latest_path}" ]]; then
    break
  fi
  if ! kill -0 "${app_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "${app_pid}" 2>/dev/null; then
  kill "${app_pid}" 2>/dev/null || true
  wait "${app_pid}" 2>/dev/null || true
fi

cat "${app_log}"

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_list}"

mapfile -t new_files < <(comm -13 "${before_list}" "${after_list}")
if [[ "${#new_files[@]}" -eq 0 ]]; then
  echo "no new .rgp files found in /tmp" >&2
  exit 1
fi

latest_path=""
latest_size=-1
for name in "${new_files[@]}"; do
  path="/tmp/${name}"
  [[ -f "${path}" ]] || continue
  size="$(stat -c '%s' "${path}")"
  if (( size > latest_size )); then
    latest_size="${size}"
    latest_path="${path}"
  fi
done

[[ -n "${latest_path}" ]] || { echo "failed to choose a capture file" >&2; exit 1; }
cp "${latest_path}" "${capture_dir}/latest.rgp"
echo "captured ${capture_dir}/latest.rgp size_bytes=${latest_size}"
