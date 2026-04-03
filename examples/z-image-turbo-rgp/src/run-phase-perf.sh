#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

phase="$(normalize_phase_name "${1:-}")"
if [[ -z "${phase}" ]]; then
    echo "usage: $(basename "$0") <condition|diffusion> [prompt...]" >&2
    exit 2
fi
shift || true

prompt="${*:-a red robot reading a book in a library}"
mkdir -p "${example_root}/perf-logs" "${output_dir}"

timestamp="$(date +%Y%m%d-%H%M%S)"
log_path="${example_root}/perf-logs/z-image-${phase}-perf-${timestamp}.log"

rm -rf "${phase_dir}"
mkdir -p "${phase_dir}"

prev_phase="${zimage_stop_after_phase}"
zimage_stop_after_phase="${phase}"
mapfile -t generation_args < <(build_generation_args)
zimage_stop_after_phase="${prev_phase}"
mapfile -t load_args < <(build_load_args)

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
echo "phase=${phase}"
print_zimage_runtime_env_summary
exit "${status}"
