#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"

run_condition_probe() {
    local before_file
    before_file="$(mktemp)"
    trap 'rm -f "${before_file}"' RETURN

    find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort > "${before_file}"

    (
        cd "${example_root}"
        env \
            ZIMAGE_STOP_AFTER_PHASE=condition \
            GGML_VK_DEBUG_MARKERS=1 \
            ZIMAGE_PHASE_MARKER_DIR="${phase_dir}" \
            MESA_VK_TRACE=rgp \
            MESA_VK_TRACE_PER_SUBMIT=1 \
            RADV_THREAD_TRACE_BUFFER_SIZE="${zimage_trace_buffer_bytes}" \
            RADV_CACHE_COUNTERS_BUFFER_SIZE="${zimage_counter_buffer_bytes}" \
            GGML_VK_PROFILE_NODES_PER_SUBMIT="${zimage_profile_nodes_per_submit}" \
            GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT="${zimage_profile_matmul_bytes_per_submit}" \
            "${app_binary}" \
            --steps "${ZIMAGE_STEPS:-2}" \
            --width "${ZIMAGE_WIDTH:-256}" \
            --height "${ZIMAGE_HEIGHT:-256}" \
            --repeat 1 \
            --seed "${ZIMAGE_SEED:-123456}" \
            --stop-after-phase condition \
            "${prompt}"
    ) >/tmp/zimage-phase-condition-probe.log 2>&1
    local status=$?

    mapfile -t new_files < <(
        comm -13 \
            "${before_file}" \
            <(find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort)
    )

    echo "phase_probe:"
    echo "  condition:"
    echo "    exit_status: ${status}"
    echo "    new_rgp_files: ${#new_files[@]}"
    if ((${#new_files[@]} > 0)); then
        printf '    captures:\n'
        for name in "${new_files[@]}"; do
            echo "      - ${name}"
        done
    fi
}

run_diffusion_probe() {
    local log_file="/tmp/zimage-phase-diffusion-probe.log"
    local status=0
    (
        cd "${example_root}"
        env \
            ZIMAGE_STOP_AFTER_PHASE=diffusion \
            ZIMAGE_STEPS="${ZIMAGE_STEPS:-2}" \
            ZIMAGE_WIDTH="${ZIMAGE_WIDTH:-256}" \
            ZIMAGE_HEIGHT="${ZIMAGE_HEIGHT:-256}" \
            ZIMAGE_REPEAT=1 \
            bash "${script_dir}/run.sh" "${prompt}"
    ) >"${log_file}" 2>&1 || status=$?

    echo "  diffusion:"
    echo "    exit_status: ${status}"
    echo "    log_tail:"
    tail -n 20 "${log_file}" | sed 's/^/      /'
}

run_diffusion_profile_probe() {
    local before_file
    before_file="$(mktemp)"
    trap 'rm -f "${before_file}"' RETURN

    find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort > "${before_file}"

    local log_file="/tmp/zimage-phase-diffusion-profile-probe.log"
    local status=0
    (
        cd "${example_root}"
        env \
            ZIMAGE_PROFILE_MODE=rgp_profile \
            ZIMAGE_STOP_AFTER_PHASE=diffusion \
            ZIMAGE_STEPS="${ZIMAGE_STEPS:-2}" \
            ZIMAGE_WIDTH="${ZIMAGE_WIDTH:-256}" \
            ZIMAGE_HEIGHT="${ZIMAGE_HEIGHT:-256}" \
            ZIMAGE_REPEAT=1 \
            MESA_VK_TRACE=rgp \
            MESA_VK_TRACE_PER_SUBMIT=1 \
            bash "${script_dir}/run.sh" "${prompt}"
    ) >"${log_file}" 2>&1 || status=$?

    mapfile -t new_files < <(
        comm -13 \
            "${before_file}" \
            <(find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort)
    )

    echo "  diffusion_profile:"
    echo "    exit_status: ${status}"
    echo "    new_rgp_files: ${#new_files[@]}"
    if ((${#new_files[@]} > 0)); then
        printf '    captures:\n'
        for name in "${new_files[@]}"; do
            echo "      - ${name}"
        done
    fi
    echo "    log_tail:"
    tail -n 20 "${log_file}" | sed 's/^/      /'
}

rm -rf "${phase_dir}"
mkdir -p "${phase_dir}"

run_condition_probe
run_diffusion_probe
run_diffusion_profile_probe
