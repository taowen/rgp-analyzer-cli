#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"
results_dir="${example_root}/perf-logs/runtime-knob-probes"
mkdir -p "${results_dir}"

declare -A CASE_TOTAL_US
declare -A CASE_TOP_FAMILY
declare -A CASE_TOP_FAMILY_TOTAL_US

bash "${script_dir}/build.sh" >/dev/null

should_run_case() {
    local name="$1"
    local filter="${ZIMAGE_RUNTIME_KNOB_CASES:-}"
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

    local log_path json_path
    log_path="${results_dir}/${name}.log"
    json_path="${results_dir}/${name}.json"

    echo "[${name}]"
    (
        cd "${example_root}"
        env \
            ZIMAGE_STEPS="${ZIMAGE_STEPS:-2}" \
            ZIMAGE_WIDTH="${ZIMAGE_WIDTH:-256}" \
            ZIMAGE_HEIGHT="${ZIMAGE_HEIGHT:-256}" \
            ZIMAGE_REPEAT=1 \
            "$@" \
            bash "${script_dir}/run-phase-perf.sh" diffusion "${prompt}"
    ) >"${log_path}" 2>&1

    python3 "${script_dir}/summarize-vk-timings.py" "${log_path}" --json > "${json_path}"
    IFS='|' read -r total_us top_family top_family_total_us < <(
        python3 - <<'PY' "${json_path}"
import json, sys
payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
families = payload.get("families") or []
top_family = families[0] if families else {}
print(
    f"{float(payload.get('total_time_us') or 0):.1f}|"
    f"{str(top_family.get('family') or '-')}|"
    f"{float(top_family.get('total_us') or 0):.1f}"
)
PY
    )
    CASE_TOTAL_US["${name}"]="${total_us}"
    CASE_TOP_FAMILY["${name}"]="${top_family}"
    CASE_TOP_FAMILY_TOTAL_US["${name}"]="${top_family_total_us}"

    python3 - <<'PY' "${name}" "${json_path}"
import json, sys
name, json_path = sys.argv[1:]
payload = json.loads(open(json_path, "r", encoding="utf-8").read())
rows = payload.get("rows") or []
families = payload.get("families") or []
top_row = rows[0] if rows else {}
top_family = families[0] if families else {}
print(
    f"  total_us={float(payload.get('total_time_us') or 0):.1f} "
    f"top_op={top_row.get('name')} "
    f"top_op_total_us={float(top_row.get('total_us') or 0):.1f}"
)
print(
    f"  top_family={top_family.get('family')} "
    f"top_family_total_us={float(top_family.get('total_us') or 0):.1f} "
    f"top_family_count={int(top_family.get('count') or 0)}"
)
PY
    echo "  log=${log_path}"
    echo
}

run_case baseline
run_case flash_attn ZIMAGE_FLASH_ATTN=1
run_case diffusion_conv_direct ZIMAGE_DIFFUSION_CONV_DIRECT=1
run_case flash_attn_conv_direct ZIMAGE_FLASH_ATTN=1 ZIMAGE_DIFFUSION_CONV_DIRECT=1
run_case clip_gpu ZIMAGE_CLIP_CPU=0
run_case disable_f16 GGML_VK_DISABLE_F16=1
run_case force_k_quant_medium GGML_VK_FORCE_K_QUANT_SHADER_SIZE=1
run_case force_k_quant_small GGML_VK_FORCE_K_QUANT_SHADER_SIZE=0
run_case force_split_k_2 GGML_VK_FORCE_SPLIT_K=2
run_case force_split_k_3 GGML_VK_FORCE_SPLIT_K=3
run_case force_split_k_4 GGML_VK_FORCE_SPLIT_K=4

if [[ -n "${CASE_TOTAL_US[baseline]:-}" ]]; then
    echo "baseline_deltas:"
    for name in baseline flash_attn diffusion_conv_direct flash_attn_conv_direct clip_gpu disable_f16 force_k_quant_medium force_k_quant_small force_split_k_2 force_split_k_3 force_split_k_4; do
        [[ -n "${CASE_TOTAL_US[${name}]:-}" ]] || continue
        python3 - <<'PY' "${name}" "${CASE_TOTAL_US[baseline]}" "${CASE_TOTAL_US[${name}]}" "${CASE_TOP_FAMILY[${name}]}" "${CASE_TOP_FAMILY_TOTAL_US[${name}]}"
import sys
name, baseline_s, current_s, family, family_total_s = sys.argv[1:]
baseline = float(baseline_s)
current = float(current_s)
delta = current - baseline
delta_pct = (delta / baseline * 100.0) if baseline else 0.0
print(
    f"  {name}: total_us={current:.1f} delta_us={delta:+.1f} delta_pct={delta_pct:+.2f}% "
    f"top_family={family} top_family_total_us={float(family_total_s):.1f}"
)
PY
    done
fi
