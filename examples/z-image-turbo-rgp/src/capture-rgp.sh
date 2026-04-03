#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash "${script_dir}/build.sh"
mkdir -p "${capture_dir}"
rm -rf "${phase_dir}"
mkdir -p "${phase_dir}"

before_file="$(mktemp)"
after_file="$(mktemp)"
before_state_file="$(mktemp)"
manifest_file="${capture_dir}/last-capture-manifest.txt"
trace_buffer_bytes="${RADV_THREAD_TRACE_BUFFER_SIZE:-${zimage_trace_buffer_bytes}}"
counter_buffer_bytes="${RADV_CACHE_COUNTERS_BUFFER_SIZE:-${zimage_counter_buffer_bytes}}"
prompt="${*:-a red robot reading a book in a library}"
trace_mode="${TRACE_MODE:-per_submit}"
mapfile -t generation_args < <(build_generation_args)
mapfile -t load_args < <(build_load_args)
app_status=0
mesa_trigger="$(mktemp /tmp/rgp-trigger.XXXXXX)"
ready_file="$(mktemp /tmp/zimage-ready.XXXXXX)"
wait_file="$(mktemp /tmp/zimage-go.XXXXXX)"
generate_ready_file="$(mktemp /tmp/zimage-generate-ready.XXXXXX)"
generate_wait_file="$(mktemp /tmp/zimage-generate-go.XXXXXX)"
rm -f "${mesa_trigger}" "${ready_file}" "${wait_file}" "${generate_ready_file}" "${generate_wait_file}"

capture_watch_dir="$(mktemp -d /tmp/zimage-rgp-watch.XXXXXX)"
capture_watch_state="$(mktemp /tmp/zimage-rgp-watch-state.XXXXXX)"
rm -f "${capture_watch_state}"
touch "${capture_watch_state}"
watcher_pid=""

cleanup() {
    if [[ -n "${watcher_pid}" ]]; then
        kill "${watcher_pid}" 2>/dev/null || true
    fi
    rm -f "${before_file}" "${after_file}" "${before_state_file}" \
        "${mesa_trigger}" "${ready_file}" "${wait_file}" "${generate_ready_file}" "${generate_wait_file}" \
        "${capture_watch_state}"
    rm -rf "${capture_watch_dir}"
}
trap cleanup EXIT

snapshot_rgp_captures() {
    local seq=0
    while [[ -f "${capture_watch_state}" ]]; do
        for source_path in /tmp/direct-txt2img_*.rgp; do
            [[ -f "${source_path}" ]] || continue
            local source_name
            source_name="$(basename "${source_path}")"
            local size_bytes
            size_bytes="$(stat -c %s "${source_path}" 2>/dev/null || echo 0)"
            local mtime_ns
            mtime_ns="$(python3 - <<'PY' "${source_path}" 2>/dev/null
import os, sys
print(os.stat(sys.argv[1]).st_mtime_ns)
PY
)"
            local key="${source_name}|${size_bytes}|${mtime_ns}"
            if grep -Fqx "${key}" "${capture_watch_state}" 2>/dev/null; then
                continue
            fi
            printf '%s\n' "${key}" >> "${capture_watch_state}"
            local snapshot_name="${source_name%.rgp}__snap$(printf '%04d' "${seq}").rgp"
            cp -p "${source_path}" "${capture_watch_dir}/${snapshot_name}" 2>/dev/null || true
            seq=$((seq + 1))
        done
        sleep 0.1
    done
}

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${before_file}"
cp "${before_file}" "${before_state_file}"
snapshot_rgp_captures &
watcher_pid=$!

mkdir -p "${output_dir}"
cd "${example_root}"
apply_zimage_runtime_env
trace_buffer_bytes="${RADV_THREAD_TRACE_BUFFER_SIZE}"
counter_buffer_bytes="${RADV_CACHE_COUNTERS_BUFFER_SIZE}"
if [[ "${trace_mode}" == "trigger" ]]; then
    echo "TRACE_MODE=trigger is not reliable for this compute-only workload; falling back to per_submit." >&2
    trace_mode="per_submit"
fi

if [[ "${trace_mode}" == "per_submit" ]]; then
    MANGOHUD=0 \
    GGML_VK_DEBUG_MARKERS="${GGML_VK_DEBUG_MARKERS}" \
    GGML_VK_PROFILE_NODES_PER_SUBMIT="${GGML_VK_PROFILE_NODES_PER_SUBMIT}" \
    GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT="${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT}" \
    GGML_VK_DISABLE_FUSION="${GGML_VK_DISABLE_FUSION:-}" \
    GGML_VK_DISABLE_GRAPH_OPTIMIZE="${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-}" \
    ZIMAGE_PHASE_MARKER_DIR="${ZIMAGE_PHASE_MARKER_DIR}" \
    MESA_VK_TRACE=rgp \
    MESA_VK_TRACE_PER_SUBMIT=1 \
    RADV_THREAD_TRACE_BUFFER_SIZE="${trace_buffer_bytes}" \
    RADV_CACHE_COUNTERS_BUFFER_SIZE="${counter_buffer_bytes}" \
        "${app_binary}" "${load_args[@]}" "${generation_args[@]}" "${prompt}" || app_status=$?
else
    MANGOHUD=0 \
    GGML_VK_DEBUG_MARKERS="${GGML_VK_DEBUG_MARKERS}" \
    GGML_VK_PROFILE_NODES_PER_SUBMIT="${GGML_VK_PROFILE_NODES_PER_SUBMIT}" \
    GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT="${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT}" \
    GGML_VK_DISABLE_FUSION="${GGML_VK_DISABLE_FUSION:-}" \
    GGML_VK_DISABLE_GRAPH_OPTIMIZE="${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-}" \
    ZIMAGE_PHASE_MARKER_DIR="${ZIMAGE_PHASE_MARKER_DIR}" \
    MESA_VK_TRACE=rgp \
    MESA_VK_TRACE_TRIGGER="${mesa_trigger}" \
    RADV_THREAD_TRACE_BUFFER_SIZE="${trace_buffer_bytes}" \
    RADV_CACHE_COUNTERS_BUFFER_SIZE="${counter_buffer_bytes}" \
        "${app_binary}" "${load_args[@]}" "${generation_args[@]}" --ready-file "${ready_file}" --generate-ready-file "${generate_ready_file}" --generate-wait-file "${generate_wait_file}" "${prompt}" &
    app_pid=$!

    while [[ ! -f "${ready_file}" ]]; do
        sleep 0.1
    done

    while [[ ! -f "${generate_ready_file}" ]]; do
        sleep 0.1
    done

    touch "${mesa_trigger}"
    sleep "${TRACE_TRIGGER_DELAY_SECONDS:-0.5}"
    touch "${generate_wait_file}"
    wait "${app_pid}" || app_status=$?
fi

if [[ -n "${watcher_pid}" ]]; then
    rm -f "${capture_watch_state}"
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
fi

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_file}"
mapfile -t new_files < <(comm -13 "${before_file}" "${after_file}")
mapfile -t watched_snapshots < <(find "${capture_watch_dir}" -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort)

if [[ "${#new_files[@]}" -eq 0 && "${#watched_snapshots[@]}" -eq 0 ]]; then
    echo "no new direct-txt2img .rgp capture was produced for TRACE_MODE=${trace_mode}" >&2
    if (( app_status != 0 )); then
        exit "${app_status}"
    fi
    exit 1
fi

largest_source=""
largest_size=0
{
    echo "trace_mode=${trace_mode}"
    echo "profile_mode=${zimage_profile_mode}"
    echo "stop_after_phase=${zimage_stop_after_phase:-full}"
    echo "app_status=${app_status}"
    echo "disable_fusion=${GGML_VK_DISABLE_FUSION:-0}"
    echo "disable_graph_optimize=${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-0}"
    echo "nodes_per_submit=${GGML_VK_PROFILE_NODES_PER_SUBMIT:-${zimage_profile_nodes_per_submit}}"
    echo "matmul_bytes_per_submit=${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-${zimage_profile_matmul_bytes_per_submit}}"
    echo "force_split_k=${GGML_VK_FORCE_SPLIT_K:-}"
    echo "generation_config=steps=${zimage_steps} width=${zimage_width} height=${zimage_height} repeat=${zimage_repeat} seed=${zimage_seed}"
    echo "phase_dir=${phase_dir}"
    echo "phase_markers:"
    for phase_file in "${phase_dir}"/*.txt; do
        [[ -f "${phase_file}" ]] || continue
        phase_name="$(basename "${phase_file}" .txt)"
        phase_value="$(cat "${phase_file}")"
        echo "  ${phase_name}=${phase_value}"
    done
    echo "captures:"
} > "${manifest_file}"

for snapshot_name in "${watched_snapshots[@]}"; do
    base_name="${snapshot_name%%__snap*}.rgp"
    if grep -Fqx "${base_name}" "${before_state_file}" 2>/dev/null; then
        continue
    fi
    source_path="${capture_watch_dir}/${snapshot_name}"
    [[ -f "${source_path}" ]] || continue
    size_bytes="$(stat -c %s "${source_path}")"
    cp -p "${source_path}" "${capture_dir}/"
    echo "  ${snapshot_name} size_bytes=${size_bytes} source=watch_snapshot" >> "${manifest_file}"
    if (( size_bytes > largest_size )); then
        largest_size="${size_bytes}"
        largest_source="${source_path}"
    fi
done

for name in "${new_files[@]}"; do
    source_path="/tmp/${name}"
    if [[ ! -f "${source_path}" ]]; then
        continue
    fi
    size_bytes="$(stat -c %s "${source_path}")"
    cp -p "${source_path}" "${capture_dir}/"
    echo "  ${name} size_bytes=${size_bytes} source=final_tmp" >> "${manifest_file}"
    if (( size_bytes > largest_size )); then
        largest_size="${size_bytes}"
        largest_source="${source_path}"
    fi
done

if [[ -z "${largest_source}" || ! -f "${largest_source}" ]]; then
    echo "new .rgp files were detected but could not be copied" >&2
    exit 1
fi

cp -p "${largest_source}" "${capture_dir}/latest.rgp"
echo "captured ${#new_files[@]} trace file(s)"
echo "selected largest capture: ${largest_source} (${largest_size} bytes)"
echo "copied to ${capture_dir}/latest.rgp"
echo "trace buffer bytes: ${trace_buffer_bytes}"
echo "counter buffer bytes: ${counter_buffer_bytes}"
echo "profile nodes per submit: ${GGML_VK_PROFILE_NODES_PER_SUBMIT:-${zimage_profile_nodes_per_submit}}"
echo "profile matmul bytes per submit: ${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-${zimage_profile_matmul_bytes_per_submit}}"
echo "force split_k: ${GGML_VK_FORCE_SPLIT_K:-}"
echo "profile mode: ${zimage_profile_mode}"
echo "disable fusion: ${GGML_VK_DISABLE_FUSION:-0}"
echo "disable graph optimize: ${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-0}"
echo "trace mode: ${trace_mode}"
echo "generation config: steps=${zimage_steps} width=${zimage_width} height=${zimage_height} repeat=${zimage_repeat} seed=${zimage_seed}"
echo "manifest: ${manifest_file}"
echo "phase markers: ${phase_dir}"
if (( app_status != 0 )); then
    echo "warning: app exited with status ${app_status}, but capture artifacts were preserved" >&2
    if [[ "${zimage_profile_mode}" != "rgp_profile" ]]; then
        exit "${app_status}"
    fi
fi
