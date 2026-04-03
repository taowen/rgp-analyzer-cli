#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"
results_dir="${example_root}/captures/profile-knob-probes"
case_timeout="${ZIMAGE_PROBE_TIMEOUT_SECONDS:-90}"
mkdir -p "${results_dir}"

bash "${script_dir}/build.sh" >/dev/null

should_run_case() {
    local name="$1"
    local filter="${ZIMAGE_PROBE_CASES:-}"
    if [[ -z "${filter}" ]]; then
        return 0
    fi
    local item
    IFS=',' read -r -a items <<< "${filter}"
    for item in "${items[@]}"; do
        if [[ "${item}" == "${name}" ]]; then
            return 0
        fi
    done
    return 1
}

run_case() {
    local name="$1"
    shift

    if ! should_run_case "${name}"; then
        return 0
    fi

    local before_file after_file log_file capture_file triage_json
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    log_file="${results_dir}/${name}.log"
    capture_file="${results_dir}/${name}.rgp"
    triage_json="${results_dir}/${name}.triage.json"

    find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort > "${before_file}"

    set +e
    (
        cd "${example_root}"
        env \
            GGML_VK_DEBUG_MARKERS=1 \
            ZIMAGE_PHASE_MARKER_DIR="${phase_dir}" \
            MESA_VK_TRACE=rgp \
            MESA_VK_TRACE_PER_SUBMIT=1 \
            RADV_THREAD_TRACE_BUFFER_SIZE="${zimage_trace_buffer_bytes}" \
            RADV_CACHE_COUNTERS_BUFFER_SIZE="${zimage_counter_buffer_bytes}" \
            "$@" \
            timeout --signal=TERM "${case_timeout}s" \
            "${app_binary}" \
            --steps "${ZIMAGE_STEPS:-2}" \
            --width "${ZIMAGE_WIDTH:-256}" \
            --height "${ZIMAGE_HEIGHT:-256}" \
            --repeat 1 \
            --seed "${ZIMAGE_SEED:-123456}" \
            --stop-after-phase diffusion \
            "${prompt}"
    ) >"${log_file}" 2>&1
    local app_status=$?
    set -e

    find /tmp -maxdepth 1 -type f -name 'direct-txt2img_*.rgp' -printf '%f\n' | sort > "${after_file}"
    mapfile -t new_files < <(comm -13 "${before_file}" "${after_file}")

    local largest=""
    local largest_size=0
    for f in "${new_files[@]}"; do
        local path="/tmp/${f}"
        [[ -f "${path}" ]] || continue
        local size
        size="$(stat -c %s "${path}")"
        if (( size > largest_size )); then
            largest_size="${size}"
            largest="${path}"
        fi
    done

    if [[ -n "${largest}" ]]; then
        cp -p "${largest}" "${capture_file}"
        env PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage "${capture_file}" --build-helper --json > "${triage_json}"
        python3 - <<'PY' "${name}" "${app_status}" "${capture_file}" "${triage_json}"
import json, sys
name, app_status, capture_path, triage_path = sys.argv[1:]
triage = json.loads(open(triage_path, "r", encoding="utf-8").read())
summary = triage.get("summary") or {}
quality = summary.get("trace_quality") or {}
decoder = summary.get("decoder") or {}
runtime = summary.get("runtime") or {}
print(f"{name}: status={app_status} capture={capture_path}")
print(
    "  trace_quality: "
    f"level={quality.get('runtime_evidence_level')} "
    f"sqtt_bytes={quality.get('sqtt_trace_bytes')} "
    f"instructions={quality.get('decoded_instruction_count')} "
    f"dispatch_spans={quality.get('dispatch_span_count')} "
    f"mapped_dispatch={quality.get('mapped_dispatch_count')}/{quality.get('total_dispatch_count')}"
)
print(
    "  runtime: "
    f"VALU={runtime.get('category_counts', {}).get('VALU')} "
    f"SALU={runtime.get('category_counts', {}).get('SALU')} "
    f"avg_stall={runtime.get('average_stall_per_instruction')} "
    f"stall_share={runtime.get('stall_share_of_duration')}"
)
print(
    "  decoder: "
    f"status={decoder.get('status')} "
    f"dispatch_isa={decoder.get('dispatch_isa_mapped')}/{decoder.get('dispatch_isa_total')} "
    f"sparse_runtime_trace={decoder.get('sparse_runtime_trace')}"
)
PY
    else
        echo "${name}: status=${app_status} capture=<none>"
        echo "  trace_quality: level=none sqtt_bytes=0 instructions=0 dispatch_spans=0 mapped_dispatch=0/0"
    fi

    echo "  log: ${log_file}"
    echo
    rm -f "${before_file}" "${after_file}"
}

run_case baseline
run_case nodes_per_submit_1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case no_fusion_submit_2 GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=2 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case no_fusion_submit_4 GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=4 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case no_fusion_submit_8 GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=8 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case no_fusion_matmul_8m GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=8388608
run_case no_fusion_submit_1 GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case disable_fusion GGML_VK_DISABLE_FUSION=1
run_case no_graph_opt_submit_1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
run_case disable_graph_opt GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
run_case no_fusion_no_graph_opt GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
run_case full_profile GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT=0
