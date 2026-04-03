#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash "${script_dir}/build.sh"
mkdir -p "${example_root}/perf-logs" "${output_dir}"

timestamp="$(date +%Y%m%d-%H%M%S)"
log_path="${example_root}/perf-logs/z-image-direct-perf-${timestamp}.log"
prompt="${*:-a red robot reading a book in a library}"
mapfile -t generation_args < <(build_generation_args)
mapfile -t load_args < <(build_load_args)

rm -rf "${phase_dir}"
mkdir -p "${phase_dir}"
apply_zimage_runtime_env
export GGML_VK_PERF_LOGGER="${GGML_VK_PERF_LOGGER:-1}"
export GGML_VK_PERF_LOGGER_CONCURRENT="${GGML_VK_PERF_LOGGER_CONCURRENT:-1}"
export GGML_VK_PERF_LOGGER_FREQUENCY="${GGML_VK_PERF_LOGGER_FREQUENCY:-1}"

cd "${example_root}"
"${app_binary}" "${load_args[@]}" "${generation_args[@]}" "${prompt}" 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}

if [[ -s "${log_path}" ]]; then
    python3 "${script_dir}/summarize-vk-timings.py" "${log_path}" || true
fi

echo "PERF_LOG=${log_path}"
print_zimage_runtime_env_summary
exit "${status}"
