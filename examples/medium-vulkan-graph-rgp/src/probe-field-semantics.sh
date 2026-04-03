#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
capture_dir="${root_dir}/captures"

bash "${script_dir}/compile-shaders.sh"
bash "${script_dir}/build.sh"

baseline_capture="${capture_dir}/field-baseline.rgp"
reg_capture="${capture_dir}/field-reg-pressure.rgp"
mem_capture="${capture_dir}/field-memory-heavy.rgp"
barrier_capture="${capture_dir}/field-barrier-heavy.rgp"

capture_named() {
    local target="$1"
    shift
    bash "${script_dir}/capture-rgp.sh" "$@"
    cp -f "${capture_dir}/latest.rgp" "${target}"
}

common_args=(
    --submit-mode phase
    --graph-iterations 12
    --warmup 1
    --dispatches-per-phase 6
    --barrier-every 2
)

capture_named "${baseline_capture}" "${common_args[@]}"

capture_named "${reg_capture}" \
    --preprocess "${script_dir}/build/shaders/preprocess_reg_pressure.spv" \
    "${common_args[@]}"

capture_named "${mem_capture}" \
    --mix "${script_dir}/build/shaders/mix_memory_heavy.spv" \
    "${common_args[@]}"

capture_named "${barrier_capture}" \
    --reduce "${script_dir}/build/shaders/reduce_barrier_heavy.spv" \
    "${common_args[@]}"

echo
echo "== baseline =="
PYTHONPATH=src python3 -m rgp_analyzer_cli shader-focus "${baseline_capture}" --code-object-index 0 --no-cache
echo
echo "== reg-pressure delta =="
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline_capture}" "${reg_capture}" --code-object-index 0 --no-cache
echo
echo "== memory-heavy delta =="
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline_capture}" "${mem_capture}" --code-object-index 1 --no-cache
echo
echo "== barrier-heavy delta =="
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline_capture}" "${barrier_capture}" --code-object-index 2 --no-cache
