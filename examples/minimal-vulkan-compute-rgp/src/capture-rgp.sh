#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
capture_dir="$(cd "${script_dir}/.." && pwd)/captures"
shader_name="${1:-reg_pressure}"
shift || true

mkdir -p "${capture_dir}"
before_file="$(mktemp)"
after_file="$(mktemp)"
trap 'rm -f "${before_file}" "${after_file}"' EXIT

trace_buffer_bytes="${RADV_THREAD_TRACE_BUFFER_SIZE:-67108864}"

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${before_file}"

MANGOHUD=0 MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE="${trace_buffer_bytes}" \
    bash "${script_dir}/run.sh" "${shader_name}" "$@"

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_file}"
newest="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
if [[ -z "${newest}" ]]; then
    newest="$(ls -1t /tmp/*.rgp | head -n 1)"
else
    newest="/tmp/${newest}"
fi

cp "${newest}" "${capture_dir}/"
cp "${newest}" "${capture_dir}/latest.rgp"
echo "captured ${newest}"
echo "copied to ${capture_dir}/latest.rgp"
echo "trace buffer bytes: ${trace_buffer_bytes}"
