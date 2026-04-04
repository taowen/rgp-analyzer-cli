#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
captures_dir="${chapter_dir}/captures"
mkdir -p "${captures_dir}"
before="$(mktemp)"; after="$(mktemp)"
trap 'rm -f "${before}" "${after}"' EXIT
find /tmp -maxdepth 1 -name '*.rgp' -printf '%f\n' | sort > "${before}" || true
MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 bash "${script_dir}/run-flash-attention.sh" "${1:-64}" >/tmp/ch08-flash-attention.log 2>&1
find /tmp -maxdepth 1 -name '*.rgp' -printf '%f\n' | sort > "${after}" || true
new_file="$(comm -13 "${before}" "${after}" | tail -n 1)"
[[ -n "${new_file}" ]] || new_file="$(find /tmp -maxdepth 1 -name '*.rgp' -printf '%T@ %p\n' | sort -n | tail -n 1 | awk '{print $2}')"
[[ "${new_file}" == /* ]] || new_file="/tmp/${new_file}"
cp "${new_file}" "${captures_dir}/flash_attention.rgp"
echo "captured ${new_file} -> ${captures_dir}/flash_attention.rgp"
